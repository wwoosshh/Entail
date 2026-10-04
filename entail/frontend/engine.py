"""frontend.engine: a small serving core written with layer B (ROADMAP M21.2).

The parts of a serving engine where the execution side's meanings live - a key/value pool in blocks, the block table,
each token's slot, continuous batching with chunked prefill, a prefix cache, a CPU tier the cache offloads to and
loads back from, preemption - written with layer B's types over Qwen3 (the model's declarations give the program's
types, as in qwen3.py). It is small on purpose: what an engine adds for speed (CUDA graphs, fused kernels, overlap of
host and device) is left out; the meanings are not.

Host side (integers keep their meaning, units.py):
  Request      a request's tokens (its prompt and what was sampled), its computed count, its blocks
  Allocator    the device pool's free blocks; full computed blocks stay cached until their room is needed
  PrefixCache  full computed blocks by the identity of their contents: a chain over the token ids from the start
               (the step program's content depends on the token ids and the positions only, check_identity)
  Engine       one step: admit (prefix hits on the device, then loads from the CPU tier), schedule (chunked prefill
               and decode), allocate (evicting cached blocks, offloading them, preempting a request when nothing is
               left), build the step's inputs, run the program, sample, advance the counts, cache what became full
The places where a count becomes another count are functions that say so (written, sampled, to_schedule, loaded):
nothing else turns a known count into a computed one.

    eng = Engine(model, blocks=64, block=16, cpu_blocks=128)
    outs = eng.generate([prompt_ids, ...], max_new=32)
"""
import hashlib
from collections import OrderedDict

import torch

from ..facts import Count, Index, Positions
from . import ops as fe
from . import paged
from . import units as U
from .graph import T, trace
from .qwen3 import _get

KNOWN, COMPUTED, SCHEDULED = Count("known"), Count("computed"), Count("scheduled")
ABS = Positions("absolute")
ROW, TOKEN = Index("row", "batch"), Index("token", "tokens")


def step(*, tokens, positions, rows, table, lengths, last, weights, pools, rotary, eps, heads, kv_heads):
    """One step of Qwen3 over a ragged list of tokens: each token has its row (sequence), its absolute position and
    its slot (through the block table); attention reads each row's computed keys; each row's last token is sampled."""
    h = fe.embed(tokens=tokens, table=weights["embed"])
    for w, pool in zip(weights["layers"], pools):
        x = fe.rms_norm(x=h, weight=w["input_norm"], eps=eps)
        q = fe.split_heads(x=fe.linear(x=x, weight=w["q"]), heads=heads, name="heads")
        k = fe.split_heads(x=fe.linear(x=x, weight=w["k"]), heads=kv_heads, name="kv_heads")
        v = fe.split_heads(x=fe.linear(x=x, weight=w["v"]), heads=kv_heads, name="kv_heads")
        q = fe.rope(x=fe.rms_norm(x=q, weight=w["q_norm"], eps=eps), positions=positions, rotary=rotary)
        k = fe.rope(x=fe.rms_norm(x=k, weight=w["k_norm"], eps=eps), positions=positions, rotary=rotary)
        at = paged.slots_of(table=table, rows=rows, positions=positions)
        keys = paged.paged_write(into=pool["keys"], src=k, at=at)
        values = paged.paged_write(into=pool["values"], src=v, at=at)
        a = paged.paged_attend(query=q, keys=keys, values=values, table=table, lengths=lengths, rows=rows,
                               at=positions, share=heads // kv_heads)
        h = fe.add(a=h, b=fe.linear(x=fe.merge_heads(x=a, name="attn"), weight=w["o"]))
        x = fe.rms_norm(x=h, weight=w["post_norm"], eps=eps)
        mlp = fe.swiglu(gate=fe.linear(x=x, weight=w["gate"]), up=fe.linear(x=x, weight=w["up"]))
        h = fe.add(a=h, b=fe.linear(x=mlp, weight=w["down"]))
    h = fe.rms_norm(x=h, weight=weights["final_norm"], eps=eps)
    logits = fe.linear(x=paged.pick(x=h, at=last), weight=weights["lm_head"])
    return {"next": fe.argmax(logits=logits)}


def copy_blocks(*, into, src, src_blocks, dst_blocks):
    """Whole blocks of every layer's keys and values from one pool to another; read only after the copy is done."""
    out = []
    for d, s in zip(into, src):
        out.append({part: paged.wait(pending=paged.move_blocks(into=d[part], src=s[part], src_blocks=src_blocks,
                                                               dst_blocks=dst_blocks))
                    for part in ("keys", "values")})
    return out


def _types(config, block, blocks, dtype, rotary):
    e, v, n = _get(config, "hidden_size"), _get(config, "vocab_size"), _get(config, "num_hidden_layers")
    hq, hkv = _get(config, "num_attention_heads"), _get(config, "num_key_value_heads")
    d, ffn = _get(config, "head_dim") or e // hq, _get(config, "intermediate_size")

    def w(dims, sizes, yields=""):
        return T(dims, dtype, "weight", sizes, (), yields)

    layer = {"input_norm": w(("embed",), (e,)), "post_norm": w(("embed",), (e,)),
             "q_norm": w(("head_dim",), (d,)), "k_norm": w(("head_dim",), (d,)),
             "q": w(("q_features", "embed"), (hq * d, e), "query"), "k": w(("kv_features", "embed"), (hkv * d, e), "key"),
             "v": w(("kv_features", "embed"), (hkv * d, e), "value"), "o": w(("embed", "attn"), (e, hq * d)),
             "gate": w(("ffn", "embed"), (ffn, e), "gate"), "up": w(("ffn", "embed"), (ffn, e), "up"),
             "down": w(("embed", "ffn"), (e, ffn))}
    pool = {"keys": paged.pool_type("key", "gpu", block, blocks, hkv, d, dtype),
            "values": paged.pool_type("value", "gpu", block, blocks, hkv, d, dtype)}
    return {"tokens": T(("tokens",), "int64", "token_ids", (None,)),
            "positions": T(("tokens",), "int64", "positions", (None,), (ABS,)),
            "rows": T(("tokens",), "int64", "row_ids", (None,), (ROW,)),
            "table": T(("batch", "blocks"), "int64", "block_table", (None, None), (Index("block", "gpu", block),)),
            "lengths": T(("batch",), "int64", "length", (None,), (COMPUTED,)),
            "last": T(("batch",), "int64", "token_index", (None,), (TOKEN,)),
            "weights": {"embed": T(("vocab", "embed"), dtype, "weight", (v, e)),
                        "layers": [dict(layer) for _ in range(n)], "final_norm": w(("embed",), (e,)),
                        "lm_head": w(("vocab", "embed"), (v, e), "logits")},
            "pools": [dict(pool) for _ in range(n)],
            "rotary": rotary, "eps": float(_get(config, "rms_norm_eps")), "heads": hq, "kv_heads": hkv}, (n, hkv, d)


def _weights(model):
    m = model.model
    layers = []
    for layer in m.layers:
        a, f = layer.self_attn, layer.mlp
        layers.append({"input_norm": layer.input_layernorm.weight, "post_norm": layer.post_attention_layernorm.weight,
                       "q_norm": a.q_norm.weight, "k_norm": a.k_norm.weight, "q": a.q_proj.weight,
                       "k": a.k_proj.weight, "v": a.v_proj.weight, "o": a.o_proj.weight,
                       "gate": f.gate_proj.weight, "up": f.up_proj.weight, "down": f.down_proj.weight})
    return {"embed": m.embed_tokens.weight, "layers": layers, "final_norm": m.norm.weight,
            "lm_head": model.lm_head.weight}


# --- the counts and where one becomes another ---------------------------------------------------------------------

@U.takes(computed=COMPUTED, scheduled=SCHEDULED)
@U.returns(COMPUTED)
def written(computed, scheduled):
    """After a step: the scheduled tokens' keys were written, so they are computed."""
    return int(computed) + int(scheduled)


@U.takes(known=KNOWN)
@U.returns(KNOWN)
def sampled(known):
    """A token was sampled: the request knows one more token (its keys are not written yet)."""
    return int(known) + 1


@U.takes(computed=COMPUTED, known=KNOWN)
@U.returns(SCHEDULED)
def to_schedule(computed, known, budget):
    """The tokens a step computes for a request: what it knows and has not computed, at most `budget`."""
    return max(0, min(int(budget), int(known) - int(computed)))


@U.takes(computed=COMPUTED)
@U.returns(COMPUTED)
def loaded(computed, blocks, block):
    """Blocks copied in (and waited for) hold keys written before: their tokens are computed."""
    return int(computed) + len(blocks) * block


# --- host side ---------------------------------------------------------------------------------------------------

class Request:
    def __init__(self, rid, prompt, max_new, eos=None):
        self.rid, self.prompt, self.max_new, self.eos = rid, list(prompt), int(max_new), eos
        self.tokens = list(prompt)
        self.blocks = []                       # Num, Index block of the device pool
        self.computed = U.num(0, COMPUTED)
        self.done = False

    @U.returns(KNOWN)
    def known(self):
        return len(self.tokens)

    @property
    def generated(self):
        return self.tokens[len(self.prompt):]


def _chain(prev, ids):
    return hashlib.sha256(prev + b"|" + ",".join(map(str, ids)).encode()).digest()


class PrefixCache:
    """Full computed blocks by the identity of their contents (a chain over the token ids from the start), on the
    device and in a CPU tier."""

    def __init__(self, block, cpu_blocks):
        self.block = block
        self.gpu = OrderedDict()               # identity -> device block (Num); order = least recently used first
        self.owner = {}                        # device block id -> identity
        self.cpu = OrderedDict()               # identity -> CPU block (Num)
        self.cpu_free = U.nums(range(cpu_blocks), Index("block", "cpu", block))

    def identities(self, ids, n_blocks):
        out, h = [], b"start"
        for i in range(n_blocks):
            h = _chain(h, ids[i * self.block:(i + 1) * self.block])
            out.append(h)
        return out

    @U.takes(upto=COMPUTED)
    def insert(self, req, upto):
        """The request's full blocks up to `upto` computed tokens become findable by their identity."""
        full = int(upto) // self.block
        for i, ident in enumerate(self.identities(req.tokens, full)):
            if ident not in self.gpu:
                blk = req.blocks[i]
                self.gpu[ident] = blk
                self.owner[int(blk)] = ident


class Allocator:
    """The device pool's blocks: free ones, cached ones (full, computed, findable by identity) with no reader, and
    the rest held by requests (counted)."""

    def __init__(self, blocks, block):
        self.free = U.nums(range(blocks), Index("block", "gpu", block))
        self.held = {}                         # device block id -> readers


class Engine:
    def __init__(self, model, blocks=64, block=16, cpu_blocks=0, max_rows=8, max_tokens=512, dtype=None, rotary=None):
        from .qwen3 import rotary_of
        self.model, self.block = model, block
        cfg = model.config
        dtype = dtype or str(model.dtype).replace("torch.", "")
        rotary = rotary or rotary_of(cfg._name_or_path, cfg)
        types, (n, hkv, d) = _types(cfg, block, blocks, dtype, rotary)
        self.program = trace(step, **types)
        paged.check_identity(self.program, covers=("tokens", "positions"))   # the cache's key: the token chain
        dev = model.device
        tdt = getattr(torch, dtype)
        self.pools = [{"keys": torch.zeros(blocks * block, hkv, d, dtype=tdt, device=dev),
                       "values": torch.zeros(blocks * block, hkv, d, dtype=tdt, device=dev)} for _ in range(n)]
        self.run = self.program.prepare(weights=_weights(model), pools=self.pools)
        self.alloc, self.cache = Allocator(blocks, block), PrefixCache(block, cpu_blocks)
        self.cpu_pools = None
        if cpu_blocks:
            pin = torch.cuda.is_available()
            self.cpu_pools = [{"keys": torch.zeros(cpu_blocks * block, hkv, d, dtype=tdt, pin_memory=pin),
                               "values": torch.zeros(cpu_blocks * block, hkv, d, dtype=tdt, pin_memory=pin)}
                              for _ in range(n)]
            gpu_t = [{k: paged.pool_type(k[:-1], "gpu", block, blocks, hkv, d, dtype) for k in ("keys", "values")}
                     for _ in range(n)]
            cpu_t = [{k: paged.pool_type(k[:-1], "cpu", block, cpu_blocks, hkv, d, dtype) for k in ("keys", "values")}
                     for _ in range(n)]
            ids = lambda pool: T(("pairs",), "int64", "block_ids", (None,), (Index("block", pool, block),))  # noqa: E731
            self.store = trace(copy_blocks, into=cpu_t, src=gpu_t, src_blocks=ids("gpu"), dst_blocks=ids("cpu")
                               ).prepare(into=self.cpu_pools, src=self.pools)
            self.load = trace(copy_blocks, into=gpu_t, src=cpu_t, src_blocks=ids("cpu"), dst_blocks=ids("gpu")
                              ).prepare(into=self.pools, src=self.cpu_pools)
        self.max_rows, self.max_tokens = max_rows, max_tokens
        self.waiting, self.running = [], []
        self.stats = {"steps": 0, "prefix_hit_tokens": 0, "loaded_blocks": 0, "stored_blocks": 0, "preempted": 0}

    # --- blocks --------------------------------------------------------------------------------------------------
    def _take_blocks(self, n, keep):
        """n device blocks: free ones first, then cached ones nobody reads (offloaded to the CPU tier first)."""
        out = []
        while len(out) < n:
            if self.alloc.free:
                out.append(self.alloc.free.pop())
                continue
            victim = next((ident for ident, blk in self.cache.gpu.items()
                           if self.alloc.held.get(int(blk), 0) == 0 and int(blk) not in keep), None)
            if victim is None:
                for b in out:
                    self.alloc.free.append(b)
                return None
            blk = self.cache.gpu.pop(victim)
            self.cache.owner.pop(int(blk), None)
            if self.cpu_pools is not None and victim not in self.cache.cpu and self.cache.cpu_free:
                cblk = self.cache.cpu_free.pop()
                self.store(src_blocks=U.tensor([blk]), dst_blocks=U.tensor([cblk]))
                self.cache.cpu[victim] = cblk
                self.stats["stored_blocks"] += 1
            out.append(blk)
        return out

    def _hold(self, blocks):
        for b in blocks:
            self.alloc.held[int(b)] = self.alloc.held.get(int(b), 0) + 1

    def _release(self, req):
        for b in req.blocks:
            k = int(b)
            self.alloc.held[k] -= 1
            if self.alloc.held[k] == 0:
                del self.alloc.held[k]
                if k not in self.cache.owner:
                    self.alloc.free.append(b)
        req.blocks = []

    # --- admission -------------------------------------------------------------------------------------------
    def _admit(self, req):
        """Prefix hits on the device, then loads from the CPU tier; at least one token is left to compute."""
        usable = (len(req.tokens) - 1) // self.block
        idents = self.cache.identities(req.tokens, usable)
        hit = []
        for ident in idents:
            blk = self.cache.gpu.get(ident)
            if blk is None:
                break
            self.cache.gpu.move_to_end(ident)
            hit.append(blk)
        self._hold(hit)
        req.blocks = list(hit)
        req.computed = U.num(len(hit) * self.block, COMPUTED)
        self.stats["prefix_hit_tokens"] += len(hit) * self.block
        if self.cpu_pools is None:
            return True
        found = []
        for ident in idents[len(hit):]:
            cblk = self.cache.cpu.get(ident)
            if cblk is None:
                break
            found.append((ident, cblk))
        if found:
            dst = self._take_blocks(len(found), keep={int(b) for b in req.blocks})
            if dst is None:
                return True
            self.load(src_blocks=U.tensor([c for _, c in found]), dst_blocks=U.tensor(dst))
            for (ident, _), blk in zip(found, dst):
                self.cache.gpu[ident] = blk
                self.cache.owner[int(blk)] = ident
            self._hold(dst)
            req.blocks += dst
            req.computed = loaded(req.computed, dst, self.block)
            self.stats["loaded_blocks"] += len(dst)
        return True

    # --- a step ----------------------------------------------------------------------------------------------
    def _preempt(self):
        req = self.running.pop()
        self._release(req)
        req.computed = U.num(0, COMPUTED)       # its keys are gone: everything it knows is computed again
        self.waiting.insert(0, req)
        self.stats["preempted"] += 1

    def step(self):
        while self.waiting and len(self.running) < self.max_rows:
            req = self.waiting.pop(0)
            self._admit(req)
            self.running.append(req)
        plan, budget = [], self.max_tokens
        for req in list(self.running):
            if req not in self.running:                       # preempted while an earlier request took blocks
                continue
            sched = to_schedule(req.computed, req.known(), budget)
            if int(sched) == 0:
                continue
            need = -(-(int(req.computed) + int(sched)) // self.block) - len(req.blocks)
            if need > 0:
                got = self._take_blocks(need, keep={int(b) for b in req.blocks})
                while got is None and len(self.running) > 1 and self.running[-1] is not req:
                    self._preempt()
                    got = self._take_blocks(need, keep={int(b) for b in req.blocks})
                if got is None:
                    continue
                self._hold(got)
                req.blocks += got
            plan.append((req, sched))
            budget -= int(sched)
            if budget <= 0:
                break
        if not plan:
            return False
        tokens, positions, rows, last, table, lengths = [], [], [], [], [], []
        width = max(len(r.blocks) for r, _ in plan)
        for i, (req, sched) in enumerate(plan):
            start = int(req.computed)
            for p in range(start, start + int(sched)):
                tokens.append(req.tokens[p])
                positions.append(p)
                rows.append(U.num(i, ROW))
            last.append(U.num(len(tokens) - 1, TOKEN))
            table.append(req.blocks + [req.blocks[0]] * (width - len(req.blocks)))
            lengths.append(written(req.computed, sched))
        dev = self.model.device
        out = self.run(tokens=torch.tensor(tokens, device=dev), positions=torch.tensor(positions, device=dev),
                       rows=U.tensor(rows, device=dev), last=U.tensor(last, device=dev),
                       table=U.tensor(table, device=dev), lengths=U.tensor(lengths, device=dev))
        nxt = out["next"].tolist()
        for (req, sched), tok in zip(plan, nxt):
            req.computed = written(req.computed, sched)
            self.cache.insert(req, req.computed)
            if int(req.computed) == int(req.known()):          # the row reached its last known token: sample
                req.tokens.append(int(tok))
                if len(req.generated) >= req.max_new or (req.eos is not None and int(tok) in req.eos):
                    req.done = True
        for req in [r for r in self.running if r.done]:
            self.running.remove(req)
            self._release(req)
        self.stats["steps"] += 1
        return True

    def generate(self, prompts, max_new, eos=None):
        reqs = [Request(i, p, max_new, eos) for i, p in enumerate(prompts)]
        self.waiting += reqs
        while self.step():
            pass
        return [r.generated for r in reqs]
