"""guarantee: the required guarantee profile for one boundary, the block FP8 matmul (ROADMAP M19 L5.4a;
docs/semantic-guarantee-design.ko.md, docs/semantic-guarantee-gpu-protocol.ko.md).

ENTAIL=guarantee turns this profile on, and nothing else: the other adapters, the off/load/debug modes and their
report-and-go-on policy are not touched, and the profile is not what ENTAIL=load does. What it promises, inside its
protected scope S and under the trust it names (Plan.trusted):

  a value the consumer hands on was computed from operands whose meaning the producers issued - which activation goes
  with which per-token-group scale, which weight with which block scale, the block size, the storage and its epoch -
  and the whole output of that very call was checked against a trusted reference within a tolerance fixed before
  the run (or is the reference's own output). Otherwise nothing is handed on: the call is refused (Refused, a
  RoleError) before its result leaves the consumer.

Three kinds of evidence are kept apart in the record: the meaning contract (issues, pairing, block, epochs,
integrity), the numeric check of the whole output of this call, and nothing sampled - this profile never compares a
slice or a warm-up and passes the rest.

The pieces:
  producers   issue_activation / issue_weight: the real quantizer and the real weight processing hand over their
              outputs with an Issue each - role, pair, block or group, layout, storage, epoch, a checksum of the
              bytes as they were made, and the source (the checkpoint's declared block, the quantizer's group).
              Nothing is guessed at the consumer from shapes: a value without an issue is not admitted.
  gate        the consumer's call. Admission first (required hooks installed, issues present, roles, pairing,
              block, epochs, the supported envelope, the budget); a known mismatch whose right value the producer
              issued is repaired before dispatch (the scale the producer paired with that value, the block it
              declared), anything else refuses. Then the kernel runs, the whole output is compared with the
              reference computed from the same operands, the operands' bytes are compared with their issue-time
              checksums, and the scales must be positive and finite. A kernel output within tolerance everywhere is
              delivered (normal); otherwise the reference's output is (repaired); integrity, scale or reference
              failures refuse (blocked). One host synchronisation per call decides it.
  graphs      inside a CUDA graph capture the same checks are captured: the kernel's output, or the reference's
              when one value is beyond the tolerance, is chosen on the device, and a failed integrity or scale
              check poisons the output with NaN; after every replay the replay hook reads the sites' flags and
              refuses the replay (raises before its outputs are handed on) when one failed, and before a replay it
              refuses when a weight the graph reads was re-issued, moved or written since capture. Python is not
              assumed to run at replay: only what was captured runs there.
  records     one JSON line per decision (guarantee-<date>.jsonl in the log folder, or ENTAIL_GUARANTEE_RECORD):
              plan, contract and environment fingerprints, producers and consumer, values and epochs, the checks,
              the tolerance, the number of elements compared, the path before and after a repair, the permit and
              whether the value was delivered.

Not covered (refused, never passed): torch.compile tracing (the producers' hooks do not run in compiled code, so the
consumer sees no issue), UE8M0 scales, block sizes, dtypes and layouts outside the plan, expert parallel and
multi-GPU. A write into an operand's storage that bypasses both the producers and PyTorch's version counter is seen
by the checksum at the next call that reads it, not before.
"""
import hashlib
import itertools
import json
import os
import threading
import time
import weakref
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

from . import core

PROFILE = "block_fp8"
CONTRACT = "block_fp8_mm/1"
BOUNDARY = "guarantee:block_fp8_mm"
CONSUMER = "vllm.model_executor.layers.quantization.utils.fp8_utils:w8a8_triton_block_scaled_mm"
ACTIVATION_PRODUCER = "vllm.model_executor.layers.quantization.utils.fp8_utils:per_token_group_quant_fp8"
WEIGHT_PRODUCER = ("vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel:"
                   "Fp8BlockScaledMMLinearKernel.process_weights_after_loading")
REQUIRED_HOOKS = (ACTIVATION_PRODUCER, WEIGHT_PRODUCER, CONSUMER)
GRAPH_HOOK = "torch.cuda.graphs:CUDAGraph.capture_begin/capture_end/replay"   # required only inside a capture
ROLES = ("activation", "activation_scale", "weight", "weight_scale")
OUTCOMES = ("normal_delivered", "repaired_delivered", "blocked", "error")
_P = 2147483647            # the checksum's modulus (2^31 - 1)
_CW = 1024                 # words per checksum row


class Refused(core.RoleError):
    """The guarantee profile handed nothing on: the call (or the replay) is refused before its result leaves."""

    def __init__(self, kind: str, why: str, record: Optional[dict] = None):
        super().__init__(f"[entail guarantee] refused ({kind}): {why}")
        self.kind, self.why, self.record = kind, why, record or {}


@dataclass(frozen=True)
class Plan:
    """The protected scope, the supported envelope, the tolerance and the trust: what a permit is issued under.
    ENTAIL_GUARANTEE_PLAN names a JSON file whose keys replace these defaults (the freeze manifest's plan)."""
    name: str = "vllm-0.30-triton-dense-block-fp8"
    scope: str = ("the output of one call of vLLM 0.30's w8a8_triton_block_scaled_mm (dense, Triton), handed to the "
                  "next operation")
    contract: str = CONTRACT
    blocks: Tuple[Tuple[int, int], ...] = ((128, 128),)
    value_dtypes: Tuple[str, ...] = ("float8_e4m3fn",)
    scale_dtypes: Tuple[str, ...] = ("float32",)
    out_dtypes: Tuple[str, ...] = ("bfloat16", "float16")
    ulps: float = 1.0                  # units in the last place of the output dtype at each value (the cast)
    c_acc: float = 2.0 ** -11          # times sum over K blocks of |a_blk| |b_blk| (the accumulation)
    budget_bytes: int = 2 << 30        # transient bytes the check may hold (an engine profile run: 8192 x 19456)
    budget_ms: Optional[float] = None  # wall time one eager check may take (None: no limit)
    chunk_bytes: int = 64 << 20        # rows of the weight dequantised at once by the reference
    graphs: bool = True                # CUDA graphs captured from eager code: checks captured with the call
    trusted: Tuple[str, ...] = ("PyTorch eager operations on the device (float32 matmul with TF32 off, "
                                "elementwise and reductions)", "the CUDA runtime, driver and device memory",
                                "this checker's code", "the producers hooked as REQUIRED_HOOKS name them")

    def to_json(self) -> dict:
        d = asdict(self)
        d["blocks"] = [list(b) for b in self.blocks]
        return d

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()


def _plan_from_env() -> Plan:
    path = os.environ.get("ENTAIL_GUARANTEE_PLAN")
    if not path:
        return Plan()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw = raw.get("plan", raw)
    known = Plan.__dataclass_fields__
    kw = {k: v for k, v in raw.items() if k in known}
    for k in ("blocks",):
        if k in kw:
            kw[k] = tuple(tuple(int(x) for x in b) for b in kw[k])
    for k in ("value_dtypes", "scale_dtypes", "out_dtypes", "trusted"):
        if k in kw:
            kw[k] = tuple(kw[k])
    return Plan(**kw)


_PLAN = None


def plan() -> Plan:
    global _PLAN
    if _PLAN is None:
        _PLAN = _plan_from_env()
    return _PLAN


def set_plan(p: Optional[Plan]) -> None:
    """Replace the plan in force (None: read it again from the environment). For harnesses and tests."""
    global _PLAN
    _PLAN = p


def active() -> bool:
    return os.environ.get("ENTAIL", "off") == "guarantee"


# --- issues: what the producers hand over -----------------------------------------------------------------------------

@dataclass
class Issue:
    serial: int
    role: str
    producer: str
    pair: int                  # the serial of the issue made together with it (value <-> its scale)
    block: Tuple[int, int]     # weight: (block_n, block_k); activation: (1, group size)
    layout: str
    snap: tuple                # (device, storage pointer, data pointer, shape, stride, offset, dtype)
    epoch: int
    version: Optional[int]
    checksum: object = None    # a device scalar: the bytes as the producer made them
    partner: object = None     # a weak reference to the paired tensor
    source: dict = field(default_factory=dict)
    declared_ok: bool = True
    why: str = ""

    def brief(self) -> dict:
        return {"serial": self.serial, "role": self.role, "producer": self.producer, "pair": self.pair,
                "block": list(self.block), "layout": self.layout, "epoch": self.epoch, "version": self.version,
                "shape": list(self.snap[3]), "stride": list(self.snap[4]), "dtype": self.snap[6]}


class _ByTensor:
    """A side table keyed by a tensor's identity, emptied when the tensor is collected (tensors compare element by
    element, so a WeakKeyDictionary cannot hold them)."""

    def __init__(self):
        self._d = {}

    def get(self, t):
        e = self._d.get(id(t))
        return None if e is None or e[0]() is not t else e[1]

    def set(self, t, v):
        k = id(t)
        self._d[k] = (weakref.ref(t, lambda _r, k=k, d=self._d: d.pop(k, None)), v)

    def clear(self):
        self._d.clear()

    def __len__(self):
        return len(self._d)


_ISSUES = _ByTensor()
_PERMITS = _ByTensor()
_EPOCHS = {}               # (device, storage pointer) -> the storage's epoch
_SERIAL = itertools.count(1)
_CALLS = itertools.count(1)
_INSTALLED = set()
_REACHED = {}
_STATS = {}
_LOCK = threading.RLock()


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


def installed(hook: str) -> None:
    """An adapter put one of REQUIRED_HOOKS in place."""
    _INSTALLED.add(hook)


def uninstalled(hook: str) -> None:
    _INSTALLED.discard(hook)


def _version(t) -> Optional[int]:
    try:
        return int(t._version)
    except Exception:  # noqa: BLE001 - an inference tensor keeps no version counter
        return None


def _storage_ptr(t) -> int:
    try:
        return int(t.untyped_storage().data_ptr())
    except Exception:  # noqa: BLE001
        return int(t.data_ptr())


def _snap(t) -> tuple:
    return (str(t.device), _storage_ptr(t), int(t.data_ptr()), tuple(int(n) for n in t.shape),
            tuple(int(s) for s in t.stride()), int(t.storage_offset()), str(t.dtype).replace("torch.", ""))


def _bump(t) -> int:
    key = (str(t.device), _storage_ptr(t))
    _EPOCHS[key] = _EPOCHS.get(key, 0) + 1
    return _EPOCHS[key]


def _epoch_now(snap) -> int:
    return _EPOCHS.get((snap[0], snap[1]), 0)


_CHECK_WEIGHTS = {}        # (device, n) -> the position weights of a checksum


def _weights(n, device):
    import torch

    key = (str(device), n)
    w = _CHECK_WEIGHTS.get(key)
    if w is None:
        w = torch.arange(1, n + 1, dtype=torch.int64, device=device) % _P
        if not _capturing():      # inside a capture the graph makes them itself (its pool is not for eager use)
            _CHECK_WEIGHTS[key] = w
    return w


def checksum(t):
    """A device scalar (int64) that changes when any byte of t changes or moves: the bytes in logical order, read as
    32-bit words where they align, each weighted by its column and its row (mod 2^31 - 1). No synchronisation; it can
    be captured in a CUDA graph."""
    import torch

    x = t.detach()
    if not x.is_contiguous():
        x = x.contiguous()
    b = x.reshape(-1).view(torch.uint8)
    try:
        w = b.view(torch.int32) if b.numel() % 4 == 0 else b.to(torch.int32)
    except RuntimeError:          # a storage offset that does not align to a word
        w = b.to(torch.int32)
    w = w.to(torch.int64)
    n = int(w.numel())
    rows = max(1, -(-n // _CW))
    if rows * _CW != n:
        w = torch.cat([w, w.new_zeros(rows * _CW - n)])
    w = w.view(rows, _CW)
    per_row = (w * _weights(_CW, w.device)[None, :]).sum(dim=1) % _P
    return ((per_row * _weights(rows, w.device)) % _P).sum() % _P


def tag(t) -> Optional[Issue]:
    return _ISSUES.get(t)


def _issue(t, role, producer, block, layout, source, ok=True, why=""):
    s = next(_SERIAL)
    iss = Issue(serial=s, role=role, producer=producer, pair=0, block=tuple(int(x) for x in block), layout=layout,
                snap=_snap(t), epoch=_bump(t), version=_version(t), checksum=checksum(t), source=dict(source),
                declared_ok=ok, why=why)
    _ISSUES.set(t, iss)
    return iss


def _pair(v, iv, s, is_):
    iv.pair, is_.pair = is_.serial, iv.serial
    iv.partner, is_.partner = weakref.ref(s), weakref.ref(v)


def issue_activation(x_q, x_s, group_size: int, producer: str = ACTIVATION_PRODUCER, layout: str = "row_major",
                     source: Optional[dict] = None) -> Tuple[Issue, Issue]:
    """The activation quantizer made x_q (fp8, [..., K]) and its per-token-group scale x_s ([..., K / group]):
    issue both, paired. Called by the adapter right after the real quantizer returned."""
    with _LOCK:
        src = dict(source or {})
        src.setdefault("group_size", int(group_size))
        ok, why = True, ""
        if x_q.shape[:-1] != x_s.shape[:-1] or int(x_s.shape[-1]) != -(-int(x_q.shape[-1]) // int(group_size)):
            ok, why = False, (f"the quantizer's scale shape {tuple(x_s.shape)} is not one per token and group of "
                              f"{group_size} for values {tuple(x_q.shape)}")
        a = _issue(x_q, "activation", producer, (1, group_size), "row_major", src, ok, why)
        s = _issue(x_s, "activation_scale", producer, (1, group_size), layout, src, ok, why)
        _pair(x_q, a, x_s, s)
        _REACHED[producer] = _REACHED.get(producer, 0) + 1
        _count("activation_issues")
        return a, s


def issue_weight(w, w_s, block: Tuple[int, int], producer: str = WEIGHT_PRODUCER, source: Optional[dict] = None
                 ) -> Tuple[Issue, Issue]:
    """The weight processing left w (fp8, [N, K]) and its block scale w_s ([ceil(N / bn), ceil(K / bk)]) for the
    kernel: issue both, paired, with the block the checkpoint declared (source carries where it was read)."""
    with _LOCK:
        bn, bk = int(block[0]), int(block[1])
        ok, why = True, ""
        if w.dim() != 2 or w_s.dim() != 2:
            ok, why = False, f"weight {tuple(w.shape)} or scale {tuple(w_s.shape)} is not 2-D"
        else:
            n, k = int(w.shape[0]), int(w.shape[1])
            want = (-(-n // bn), -(-k // bk))
            if tuple(int(x) for x in w_s.shape) != want:
                ok, why = False, (f"the block scale has shape {tuple(w_s.shape)}, the declared block {bn}x{bk} over "
                                  f"the weight {n}x{k} gives {want}")
        declared = (source or {}).get("declared_block")
        if ok and declared is not None and tuple(int(x) for x in declared) != (bn, bk):
            ok, why = False, (f"the kernel was set up with block {bn}x{bk}, the checkpoint declares "
                              f"{list(declared)}")
        b = _issue(w, "weight", producer, (bn, bk), "blocked", source or {}, ok, why)
        s = _issue(w_s, "weight_scale", producer, (bn, bk), "blocked", source or {}, ok, why)
        _pair(w, b, w_s, s)
        _REACHED[producer] = _REACHED.get(producer, 0) + 1
        _count("weight_issues")
        return b, s


def _stale(t, iss: Issue) -> Optional[str]:
    """Why t no longer holds what its producer issued as far as the host can tell (storage, layout, epoch, version);
    None when it does. The bytes themselves are compared on the device (checksum)."""
    now = _snap(t)
    if now != iss.snap:
        return f"{iss.role} moved or changed its layout since its issue ({iss.snap[2:6]} -> {now[2:6]})"
    if _epoch_now(now) != iss.epoch:
        return (f"{iss.role}'s storage was issued again (epoch {iss.epoch} -> {_epoch_now(now)}): it holds another "
                f"producer value now")
    v = _version(t)
    if iss.version is not None and v is not None and v != iss.version:
        return f"{iss.role} was written in place since its issue (version {iss.version} -> {v})"
    return None


# --- records --------------------------------------------------------------------------------------------------------

_ENV = None


def environment() -> dict:
    global _ENV
    if _ENV is None:
        import platform
        import sys

        env = {"python": sys.version.split()[0], "platform": platform.platform()}
        try:
            import torch

            env["torch"] = torch.__version__
            env["cuda"] = torch.version.cuda
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(torch.cuda.current_device())
                env["gpu"] = p.name
                env["cc"] = f"{p.major}.{p.minor}"
                env["vram"] = int(p.total_memory)
        except Exception:  # noqa: BLE001
            pass
        for mod in ("triton", "vllm"):
            try:
                from importlib import metadata

                env[mod] = metadata.version(mod)
            except Exception:  # noqa: BLE001
                env[mod] = None
        try:
            from . import __version__

            env["entail"] = __version__
        except Exception:  # noqa: BLE001
            pass
        env["fingerprint"] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
        _ENV = env
    return _ENV


def _record_path() -> Optional[str]:
    p = os.environ.get("ENTAIL_GUARANTEE_RECORD")
    if p:
        return None if p.strip().lower() == "off" else p
    from . import record

    folder = record.log_dir()
    return os.path.join(folder, f"guarantee-{time.strftime('%Y-%m-%d')}.jsonl") if folder else None


def _write(line: dict) -> None:
    path = _record_path()
    if path is None:
        return
    from . import record

    full = {"v": 1, "t": round(time.time(), 3), "run": record.run_id(), "pid": os.getpid(), "profile": PROFILE}
    full.update(line)
    record._append(path, json.dumps(full, ensure_ascii=False, default=str) + "\n")


def _base(call: int) -> dict:
    p = plan()
    return {"kind": "guarantee_call", "call": call, "plan": p.name, "plan_fp": p.fingerprint(), "contract": p.contract,
            "env_fp": environment()["fingerprint"], "consumer": CONSUMER, "boundary": BOUNDARY}


def _refuse(rec: dict, kind: str, why: str, t0: float):
    rec.update({"outcome": "blocked", "blocked_kind": kind, "reason": why, "permit": None, "delivered": False,
                "ms_total": round((time.perf_counter() - t0) * 1e3, 3)})
    _count("blocked")
    _count(f"blocked_{kind}")
    _write(rec)
    raise Refused(kind, why, rec)


# --- the consumer's gate ----------------------------------------------------------------------------------------------

def _dt(t) -> str:
    return str(t.dtype).replace("torch.", "")


def _admit(A, B, As, Bs, block_size, output_dtype, rec, t0):
    """Admission and the repairs before dispatch. Returns (A, B, As, Bs, block, issues) or refuses."""
    import torch

    p = plan()
    missing = [h for h in REQUIRED_HOOKS if h not in _INSTALLED]
    if missing:
        _refuse(rec, "hook", f"required hook(s) not installed: {', '.join(missing)}", t0)
    if torch.compiler.is_compiling():
        _refuse(rec, "unsupported", "called while torch.compile traces: the profile runs eager code or CUDA graphs "
                                    "captured from it", t0)
    if "e8m0" in _dt(As) or "e8m0" in _dt(Bs):
        _refuse(rec, "unsupported", "exponent-only (UE8M0) scales are outside the plan", t0)
    tA, tAs, tB, tBs = tag(A), tag(As), tag(B), tag(Bs)
    for t, tg, role in ((A, tA, "activation"), (B, tB, "weight")):
        if tg is None:
            _refuse(rec, "declaration", f"the {role} operand ({tuple(t.shape)}, {_dt(t)}) has no producer issue: it "
                                        f"was not made by the hooked producer (a compiled path, another quantizer, "
                                        f"a copy)", t0)
        if tg.role != role:
            _refuse(rec, "contract", f"the {role} operand was issued as {tg.role}", t0)
    repairs = []
    for name, vt, gt, srole in (("As", tA, tAs, "activation_scale"), ("Bs", tB, tBs, "weight_scale")):
        if gt is not None and gt.role == srole and gt.pair == vt.serial:
            continue
        partner = vt.partner() if vt.partner is not None else None
        pt = tag(partner) if partner is not None else None
        if partner is None or pt is None or pt.pair != vt.serial:
            _refuse(rec, "pairing", f"{name} is not the scale issued with its value (issue {vt.serial}) and that "
                                    f"scale is gone", t0)
        given = "unissued" if gt is None else f"issue {gt.serial} ({gt.role}, paired with issue {gt.pair})"
        repairs.append({"handle": f"pair_{name}",
                        "why": f"{name} given was {given}; the producer paired issue {vt.serial} with issue "
                               f"{pt.serial}",
                        "before": None if gt is None else gt.serial, "after": pt.serial})
        if name == "As":
            As, tAs = partner, pt
        else:
            Bs, tBs = partner, pt
    block = tuple(int(x) for x in block_size)
    if block != tuple(tB.block):
        repairs.append({"handle": "block", "why": f"block_size {list(block)} given, the checkpoint declared "
                                                  f"{list(tB.block)} for this weight", "before": list(block),
                        "after": list(tB.block)})
        block = tuple(tB.block)
    issues = {"A": tA, "As": tAs, "B": tB, "Bs": tBs}
    rec["issues"] = {k: v.brief() for k, v in issues.items()}
    rec["repairs"] = repairs
    for k, v in issues.items():
        if not v.declared_ok:
            _refuse(rec, "declaration", f"{k}: the producer's declaration does not hold: {v.why}", t0)
    if int(tA.block[1]) != block[1]:
        _refuse(rec, "contract", f"the activation was quantized in groups of {tA.block[1]}, the weight's block_k is "
                                 f"{block[1]}", t0)
    if block not in tuple(tuple(b) for b in p.blocks):
        _refuse(rec, "unsupported", f"block {list(block)} is outside the plan ({[list(b) for b in p.blocks]})", t0)
    for t, role, allowed in ((A, "activation", p.value_dtypes), (B, "weight", p.value_dtypes),
                             (As, "activation_scale", p.scale_dtypes), (Bs, "weight_scale", p.scale_dtypes)):
        if _dt(t) not in allowed:
            _refuse(rec, "unsupported", f"{role} dtype {_dt(t)} is outside the plan ({', '.join(allowed)})", t0)
    out = str(output_dtype or torch.float16).replace("torch.", "")
    if out not in p.out_dtypes:
        _refuse(rec, "unsupported", f"output dtype {out} is outside the plan ({', '.join(p.out_dtypes)})", t0)
    for k, t in (("A", A), ("As", As), ("B", B), ("Bs", Bs)):
        why = _stale(t, issues[k])
        if why:
            _refuse(rec, "epoch", f"{k}: {why}", t0)
    K = int(A.shape[-1])
    if A.dim() < 2 or B.dim() != 2 or Bs.dim() != 2 or int(B.shape[1]) != K:
        _refuse(rec, "contract", f"shapes A {tuple(A.shape)}, B {tuple(B.shape)}, Bs {tuple(Bs.shape)} do not make "
                                 f"activation x weight^T", t0)
    if not A.is_contiguous() or K % block[1]:
        _refuse(rec, "unsupported", f"A must be contiguous with K a multiple of block_k (stride {tuple(A.stride())}, "
                                    f"K {K})", t0)
    M, N = A.numel() // K, int(B.shape[0])
    if tuple(As.shape[:-1]) != tuple(A.shape[:-1]) or int(As.shape[-1]) != K // block[1]:
        _refuse(rec, "contract", f"As {tuple(As.shape)} is not one scale per token and K group for A "
                                 f"{tuple(A.shape)}", t0)
    if tuple(int(x) for x in Bs.shape) != (-(-N // block[0]), -(-K // block[1])):
        _refuse(rec, "contract", f"Bs {tuple(Bs.shape)} is not one scale per {block[0]}x{block[1]} block of B "
                                 f"{N}x{K}", t0)
    # the dequantised activation (twice, while it is made), the reference's output, two chunks of work tensors, and
    # the checksums' 64-bit words (twice the bytes of each operand, twice while weighted)
    need = 8 * M * K + M * N * (2 if out in ("bfloat16", "float16") else 4) + 2 * p.chunk_bytes + 4 * (M * K + N * K)
    rec["budget"] = {"bytes": need, "limit": p.budget_bytes}
    if need > p.budget_bytes:
        _refuse(rec, "budget", f"the check needs about {need} transient bytes, the plan allows {p.budget_bytes}", t0)
    rec["shape"] = {"M": M, "N": N, "K": K}
    rec["block"] = list(block)
    rec["out_dtype"] = out
    return A, B, As, Bs, block, issues


class _Precision:
    """float32 matmul at full precision (TF32 off) for the reference, restored afterwards."""

    def __enter__(self):
        import torch

        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        self.prec = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    def __exit__(self, *exc):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.set_float32_matmul_precision(self.prec)


def reference(A, B, As, Bs, block):
    """(r, S): the product of the block-dequantised operands in float32 ([M, N]), and for each value the sum over
    the K blocks of |a_blk| |b_blk| (the scale the accumulation error is bounded by). Weight rows are dequantised a
    chunk at a time (plan().chunk_bytes)."""
    import torch

    gn, gk = int(block[0]), int(block[1])
    K = int(A.shape[-1])
    M, N, nb = A.numel() // K, int(B.shape[0]), K // gk
    a = A.reshape(M, nb, gk).to(torch.float32) * As.reshape(M, nb).to(torch.float32)[:, :, None]
    an = a.norm(dim=2)
    a = a.reshape(M, K)
    r = torch.empty((M, N), dtype=torch.float32, device=A.device)
    S = torch.empty((M, N), dtype=torch.float32, device=A.device)
    rows = max(gn, (plan().chunk_bytes // max(1, K * 4)) // gn * gn)
    with _Precision():
        for n0 in range(0, N, rows):
            n1 = min(N, n0 + rows)
            s = Bs[n0 // gn: -(-n1 // gn)].to(torch.float32).repeat_interleave(gn, dim=0)[: n1 - n0]
            b = B[n0:n1].to(torch.float32).reshape(n1 - n0, nb, gk) * s[:, :, None]
            r[:, n0:n1] = a @ b.reshape(n1 - n0, K).T
            S[:, n0:n1] = an @ b.norm(dim=2).T
    return r, S


def ulp(x, dtype):
    """One unit in the last place of `dtype` at |x| (float32 tensor in, float32 out); the smallest subnormal at 0."""
    import torch

    prec, tiny = {"bfloat16": (8, 2.0 ** -133), "float16": (11, 2.0 ** -24)}.get(
        str(dtype).replace("torch.", ""), (24, 2.0 ** -149))
    _m, e = torch.frexp(x.abs())
    u = torch.ldexp(torch.ones_like(x), e - prec)
    return torch.where(x == 0, torch.full_like(x, tiny), u.clamp_min(tiny))


def _columns(M, K, gn, p) -> int:
    """Weight rows (output columns) checked at once: bounded by the weight chunk (dequantised, K x 4 bytes a row)
    and by the [M, columns] float32 work tensors of the comparison; a multiple of the block."""
    by_b = p.chunk_bytes // max(1, K * 4)
    by_r = p.chunk_bytes // max(1, M * 4 * 6)
    return max(gn, min(by_b, by_r) // gn * gn)


def _check(k, A, B, As, Bs, block, out_dtype, issues):
    """On the device, no synchronisation, a chunk of output columns at a time (memory bounded by the plan's
    chunk_bytes; every value of the output is compared): (ref, status). ref: the reference's output in the output
    dtype, [M, N]. status (int64 [6]): values beyond the tolerance, reference values not finite, operands whose
    bytes differ from their issue (a bit each: A, As, B, Bs), scales not positive and finite, kernel values not
    finite, the largest ratio to the tolerance x 2^20 (rounded)."""
    import torch

    p = plan()
    gn, gk = int(block[0]), int(block[1])
    K = int(A.shape[-1])
    M, N, nb = A.numel() // K, int(B.shape[0]), K // gk
    dev = A.device
    a = A.reshape(M, nb, gk).to(torch.float32) * As.reshape(M, nb).to(torch.float32)[:, :, None]
    an = a.norm(dim=2)
    a = a.reshape(M, K)
    kk = k.reshape(M, N)
    ref = torch.empty((M, N), dtype=out_dtype, device=dev)
    z = torch.zeros((), dtype=torch.int64, device=dev)
    viol, rbad, kbad = z.clone(), z.clone(), z.clone()
    ratio = torch.zeros((), dtype=torch.float32, device=dev)
    cols = _columns(M, K, gn, p)
    with _Precision():
        for n0 in range(0, N, cols):
            n1 = min(N, n0 + cols)
            s = Bs[n0 // gn: -(-n1 // gn)].to(torch.float32).repeat_interleave(gn, dim=0)[: n1 - n0]
            b = B[n0:n1].to(torch.float32).reshape(n1 - n0, nb, gk) * s[:, :, None]
            r = a @ b.reshape(n1 - n0, K).T
            tol = p.ulps * ulp(r, out_dtype) + p.c_acc * (an @ b.norm(dim=2).T)
            del b
            kf = kk[:, n0:n1].to(torch.float32)
            diff = (kf - r).abs()
            rfin, kfin = torch.isfinite(r), torch.isfinite(kf)
            viol += (~((diff <= tol) & kfin & rfin)).sum()
            rbad += (~rfin).sum()
            kbad += (~kfin).sum()
            ratio = torch.maximum(ratio, torch.where(torch.isfinite(diff), diff / tol.clamp_min(1e-38),
                                                     torch.zeros_like(diff)).amax())
            ref[:, n0:n1] = r.to(out_dtype)
    bits = z.clone()
    for i, (x, name) in enumerate(((A, "A"), (As, "As"), (B, "B"), (Bs, "Bs"))):
        bits = bits + (checksum(x) != issues[name].checksum).to(torch.int64) * (1 << i)
    sbad = (~((As > 0) & torch.isfinite(As)).all()) | (~((Bs > 0) & torch.isfinite(Bs)).all())
    status = torch.stack([viol, rbad, bits, sbad.to(torch.int64), kbad,
                          (ratio.clamp_max(2.0 ** 40) * 2.0 ** 20).round().to(torch.int64)])
    return ref, status


def _capturing() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def gate(kernel, A, B, As, Bs, block_size, output_dtype=None, consumer: str = CONSUMER):
    """The consumer's call under the profile: admission, dispatch, the whole-output check, and the permit. Returns
    the value handed on, or raises Refused (blocked) - or the kernel's own error, recorded as an execution error."""
    import torch

    call = next(_CALLS)
    t0 = time.perf_counter()
    rec = _base(call)
    rec["consumer"] = consumer
    rec["path_before"] = "kernel"
    _count("calls")
    try:
        A, B, As, Bs, block, issues = _admit(A, B, As, Bs, block_size, output_dtype, rec, t0)
    except Refused:
        raise
    except Exception as e:  # noqa: BLE001 - the checker failed: no permit, said as the checker's fault
        _count("checker_errors")
        _refuse(rec, "checker", f"admission raised {type(e).__name__}: {e}", t0)
    out_dtype = output_dtype or torch.float16
    if A.numel() == 0:
        rec.update({"outcome": "normal_delivered", "permit": call, "delivered": True, "elements": 0,
                    "path_after": "kernel", "note": "no token: nothing to compare"})
        _write(rec)
        _count("normal_delivered")
        return kernel(A, B, As, Bs, list(block), out_dtype)
    if _capturing():
        return _captured(kernel, A, B, As, Bs, block, out_dtype, issues, rec, t0)
    t1 = time.perf_counter()
    try:
        k = kernel(A, B, As, Bs, list(block), out_dtype)
    except Exception as e:
        rec.update({"outcome": "error", "reason": f"the kernel raised {type(e).__name__}: {e}", "permit": None,
                    "delivered": False})
        _count("kernel_errors")
        _write(rec)
        raise
    t2 = time.perf_counter()
    try:
        ref, status = _check(k, A, B, As, Bs, block, out_dtype, issues)
        st = [int(x) for x in status.tolist()]
    except Exception as e:  # noqa: BLE001
        _count("checker_errors")
        _refuse(rec, "checker", f"the check raised {type(e).__name__}: {e}", t0)
    t3 = time.perf_counter()
    viol, rbad, bits, sbad, kbad, ratio = st
    p = plan()
    M, N = rec["shape"]["M"], rec["shape"]["N"]
    rec["checks"] = {"elements": M * N, "beyond_tolerance": viol, "kernel_nonfinite": kbad,
                     "reference_nonfinite": rbad, "integrity_bits": bits, "scales_bad": bool(sbad),
                     "max_ratio": ratio / 2.0 ** 20}
    rec["tolerance"] = {"ulps": p.ulps, "c_acc": p.c_acc, "form": "ulps*ulp(out,|r|) + c_acc*sum_blk|a_blk||b_blk|"}
    rec["ms_kernel"] = round((t2 - t1) * 1e3, 3)
    rec["ms_check"] = round((t3 - t2) * 1e3, 3)
    if bits:
        names = [n for i, n in enumerate(("A", "As", "B", "Bs")) if bits >> i & 1]
        _refuse(rec, "integrity", f"the bytes of {', '.join(names)} differ from what the producer issued", t0)
    if sbad:
        _refuse(rec, "scales", "a scale is not positive and finite", t0)
    if rbad:
        _refuse(rec, "reference", f"{rbad} reference values are not finite: the inputs are outside the supported "
                                  f"range", t0)
    if p.budget_ms is not None and (t3 - t0) * 1e3 > p.budget_ms:
        _refuse(rec, "budget", f"the check took {(t3 - t0) * 1e3:.3f} ms, the plan allows {p.budget_ms} ms", t0)
    if viol:
        out = ref.reshape(*A.shape[:-1], N)
        rec.update({"outcome": "repaired_delivered", "path_after": "reference",
                    "recheck": {"reference_finite": True, "shape": list(out.shape), "dtype": _dt(out)},
                    "reason": f"{viol} of {M * N} kernel values beyond the tolerance (largest ratio "
                              f"{ratio / 2.0 ** 20:.4g}); the reference's output is handed on"})
        _count("repaired_delivered")
    else:
        out = k
        rec.update({"outcome": "repaired_delivered" if rec["repairs"] else "normal_delivered",
                    "path_after": "kernel"})
        _count(rec["outcome"])
    rec.update({"permit": call, "delivered": True, "ms_total": round((time.perf_counter() - t0) * 1e3, 3)})
    _PERMITS.set(out, {"permit": call, "plan_fp": rec["plan_fp"]})
    _write(rec)
    return out


def permit_of(t) -> Optional[dict]:
    return _PERMITS.get(t)


# --- CUDA graphs ------------------------------------------------------------------------------------------------------

class _Site:
    """One gate captured in a graph: the weights it reads (checked on the host before each replay) and its flags
    (written on the device at each replay)."""

    def __init__(self, call, rec, issues, B, Bs, flags):
        self.call, self.rec, self.flags = call, rec, flags
        self.B, self.Bs = weakref.ref(B), weakref.ref(Bs)
        self.issues = issues


_GRAPHS = {}               # id(graph) -> (weakref to graph, [sites])
_CAPTURE = threading.local()


def capture_begin(graph) -> None:
    _CAPTURE.graph = graph


def capture_end(graph) -> None:
    if getattr(_CAPTURE, "graph", None) is graph:
        _CAPTURE.graph = None


def _captured(kernel, A, B, As, Bs, block, out_dtype, issues, rec, t0):
    import torch

    p = plan()
    graph = getattr(_CAPTURE, "graph", None)
    if not p.graphs:
        _refuse(rec, "unsupported", "CUDA graph capture is outside the plan", t0)
    if graph is None or GRAPH_HOOK not in _INSTALLED:
        _refuse(rec, "hook", "a capture this profile was not told about (torch.cuda.CUDAGraph.capture_begin not "
                             "hooked): its replays could not be checked", t0)
    k = kernel(A, B, As, Bs, list(block), out_dtype)
    ref, status = _check(k, A, B, As, Bs, block, out_dtype, issues)
    bad = (status[1] + status[2] + status[3]) > 0
    out = torch.where(status[0] > 0, ref.reshape(k.shape), k)   # one value beyond: all the reference
    out = torch.where(bad, torch.full_like(out, float("nan")), out)
    flags = torch.zeros(6, dtype=torch.int64, device=A.device)
    flags.copy_(status)
    rec["captured"] = True
    rec["tolerance"] = {"ulps": p.ulps, "c_acc": p.c_acc, "form": "ulps*ulp(out,|r|) + c_acc*sum_blk|a_blk||b_blk|"}
    site = _Site(rec["call"], dict(rec), issues, B, Bs, flags)
    entry = _GRAPHS.get(id(graph))
    if entry is None or entry[0]() is not graph:
        entry = _GRAPHS[id(graph)] = (weakref.ref(graph), [])
    entry[1].append(site)
    rec.update({"outcome": "captured", "path_after": "the kernel's output, or the reference's when one value is "
                                                     "beyond the tolerance, chosen on the device",
                "permit": None, "delivered": False,
                "note": "decided at each replay by the replay hook"})
    _count("captured_sites")
    _write(rec)
    return out


def before_replay(graph) -> None:
    """Refuse a replay whose graph reads a weight that was re-issued, moved or written since capture."""
    entry = _GRAPHS.get(id(graph))
    if entry is None or entry[0]() is not graph:
        return
    for site in entry[1]:
        for name, ref in (("B", site.B), ("Bs", site.Bs)):
            t = ref()
            why = "it was collected" if t is None else _stale(t, site.issues[name])
            if why:
                rec = dict(site.rec)
                rec.update({"kind": "guarantee_replay", "replay_of": site.call, "call": next(_CALLS)})
                _refuse(rec, "epoch", f"the graph reads {name}, and {why}", time.perf_counter())


def after_replay(graph) -> None:
    """Read the flags every gate in the graph wrote at this replay; refuse the replay when one failed."""
    entry = _GRAPHS.get(id(graph))
    if entry is None or entry[0]() is not graph:
        return
    import torch

    torch.cuda.current_stream().synchronize()
    flags = torch.stack([s.flags for s in entry[1]]).tolist()
    _count("replays")
    blocked = None
    for site, st in zip(entry[1], flags):
        viol, rbad, bits, sbad, kbad, ratio = (int(x) for x in st)
        rec = dict(site.rec)
        M, N = rec["shape"]["M"], rec["shape"]["N"]
        rec.update({"kind": "guarantee_replay", "replay_of": site.call, "call": next(_CALLS),
                    "checks": {"elements": M * N, "beyond_tolerance": viol, "kernel_nonfinite": kbad,
                               "reference_nonfinite": rbad, "integrity_bits": bits, "scales_bad": bool(sbad),
                               "max_ratio": ratio / 2.0 ** 20}})
        if bits or sbad or rbad:
            kind = "integrity" if bits else ("scales" if sbad else "reference")
            rec.update({"outcome": "blocked", "blocked_kind": kind, "permit": None, "delivered": False,
                        "reason": "the replay's output was poisoned on the device and the replay is refused"})
            _count("blocked")
            _count(f"blocked_{kind}")
            blocked = blocked or (kind, rec)
        elif viol:
            rec.update({"outcome": "repaired_delivered", "path_after": "reference (chosen on the device)",
                        "permit": rec["call"], "delivered": True})
            _count("repaired_delivered")
        else:
            rec.update({"outcome": "repaired_delivered" if rec.get("repairs") else "normal_delivered",
                        "path_after": "kernel", "permit": rec["call"], "delivered": True})
            _count(rec["outcome"])
        _write(rec)
    if blocked:
        raise Refused(blocked[0], "a gate in the replayed graph failed its check; the replay's outputs are refused",
                      blocked[1])


# --- state ------------------------------------------------------------------------------------------------------------

def stats() -> dict:
    out = dict(_STATS)
    out["installed"] = sorted(_INSTALLED)
    out["reached"] = dict(_REACHED)
    return out


def reset() -> None:
    """Forget every issue, permit, graph and count (tests and harnesses; not the installed hooks)."""
    _ISSUES.clear()
    _PERMITS.clear()
    _EPOCHS.clear()
    _GRAPHS.clear()
    _STATS.clear()
    _REACHED.clear()
    _CHECK_WEIGHTS.clear()
