"""vocab_contract: the tokenizer an engine holds against the model's vocabulary, in the core (LIBRARY_DESIGN.md
4.6, 4.7; ROADMAP M15.3; realworld/CODEBOOK_v2.md G; transformers#48967).

A model folder declares its vocabulary in more than one place: the tokenizer files (tokenizer.json, vocab.txt,
vocab.json, a sentencepiece model), config.json's vocab_size, and the embedding's rows in the checkpoint. The engine
builds one tokenizer from the folder by its own precedence (transformers 5 takes tokenizer.json when it is there) and
nothing compares the result with the model. When the folder carries two tokenizers - vocab.txt with 100,000 entries
that is the model's, and a tokenizer.json with 32,000 from another model - the engine can build the wrong one, and
every id it produces means another token, silently.

The facts are read here from the folder (no engine); an adapter says only what the engine's tokenizer holds
(its base vocabulary and its length). The rules are here, once, and take no threshold:

  vocab_out_of_range    the tokenizer's length (its highest id + 1) exceeds the embedding's rows: an id with no row
  vocab_not_the_models  the folder's tokenizer sources declare different vocabularies; the model's is the one that
                        agrees with its vocab_size / embedding rows, and the engine holds the other one

A tokenizer whose base size is none of the folder's sources is said to be unknown (which source it came from cannot
be told), not broken. Ordinary folders - one tokenizer source, an embedding padded past the vocabulary (Qwen 151,936
rows for 151,665 tokens) - pass: the tokenizer fits and there is nothing to disagree with. No repair is offered yet:
a decision names the source that is the model's, so the user can load from it.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import tally as _tally

EMBEDDING_SUFFIXES = ("embed_tokens.weight", "word_embeddings.weight", "wte.weight", "tok_embeddings.weight",
                      "embed.weight", "embedding.weight")


@dataclass
class Sources:
    candidates: List[Tuple[int, str]] = field(default_factory=list)   # (base size, where) per tokenizer source
    rows: Optional[int] = None            # the embedding's rows (verified) or config.json's vocab_size (declared)
    rows_where: str = ""
    rows_verified: bool = False
    problems: List[str] = field(default_factory=list)


def _spm_pieces(path) -> Optional[int]:
    """The pieces of a sentencepiece model: field 1 of its ModelProto, counted by walking the protobuf top level
    (no sentencepiece library needed). None when the bytes do not parse."""
    data = open(path, "rb").read()
    i, n, size = 0, 0, len(data)

    def varint(j):
        r, s = 0, 0
        while j < size:
            b = data[j]
            j += 1
            r |= (b & 0x7F) << s
            s += 7
            if not b & 0x80:
                return r, j
        raise ValueError("truncated varint")

    try:
        while i < size:
            tag, i = varint(i)
            fld, wt = tag >> 3, tag & 7
            if wt == 0:
                _, i = varint(i)
            elif wt == 1:
                i += 8
            elif wt == 2:
                ln, i = varint(i)
                i += ln
            elif wt == 5:
                i += 4
            else:
                return None
            if fld == 1:
                n += 1
        return n
    except (ValueError, IndexError):
        return None


def sources(path) -> Sources:
    """What the folder declares about its vocabulary."""
    s = Sources()
    p = os.path.join(path, "tokenizer.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            vocab = (d.get("model") or {}).get("vocab")
            if isinstance(vocab, (dict, list)):
                s.candidates.append((len(vocab), "tokenizer.json (model.vocab)"))
            else:
                s.problems.append("tokenizer.json: no model.vocab to count")
        except (ValueError, OSError) as e:
            s.problems.append(f"tokenizer.json: {type(e).__name__}")
    p = os.path.join(path, "vocab.txt")
    if os.path.isfile(p):
        text = open(p, encoding="utf-8", errors="replace").read()
        n = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        s.candidates.append((n, "vocab.txt"))
    p = os.path.join(path, "vocab.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            if isinstance(d, dict):
                s.candidates.append((len(d), "vocab.json"))
        except (ValueError, OSError) as e:
            s.problems.append(f"vocab.json: {type(e).__name__}")
    for name in ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model"):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            n = _spm_pieces(p)
            if n:
                s.candidates.append((n, f"{name} (sentencepiece pieces)"))
            else:
                s.problems.append(f"{name}: not read as a sentencepiece model")
    config_vocab = None
    p = os.path.join(path, "config.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            v = d.get("vocab_size") or (d.get("text_config") or {}).get("vocab_size")
            if isinstance(v, int) and v > 0:
                config_vocab = v
        except (ValueError, OSError) as e:
            s.problems.append(f"config.json: {type(e).__name__}")
    try:
        from . import observe

        ck = observe.Checkpoint(path)
        chosen = None
        for suffix in EMBEDDING_SUFFIXES:
            for name in (ck.find(suffix) if ck.tensors else []):
                shape = list(ck.tensors[name][2])
                # the token embedding has at least the config's vocabulary of rows; a position or patch embedding
                # under a similar name has far fewer (M15.3 review)
                if shape and (config_vocab is None or int(shape[0]) >= config_vocab):
                    chosen = (name, shape)
                    break
            if chosen:
                break
        if chosen:
            s.rows, s.rows_where, s.rows_verified = int(chosen[1][0]), f"{chosen[0]} {chosen[1]} in the checkpoint", True
    except Exception as e:  # noqa: BLE001 - no safetensors, or unreadable: the config stands in below
        s.problems.append(f"embedding rows not read: {type(e).__name__}")
    if s.rows is None and config_vocab is not None:
        s.rows, s.rows_where = config_vocab, "config.json vocab_size"
    return s


def check(boundary: str, consumer: str, path: str, tokenizer_size: Optional[int], tokenizer_len: Optional[int],
          where: str, policy=None, owner=None, record: bool = True) -> list:
    """Decide the tokenizer the engine holds (`tokenizer_size` base tokens, `tokenizer_len` its highest id + 1)
    against what the folder at `path` declares. Returns the decisions; with `record` they are also recorded through
    load.enforce (the run goes on, or stops where the policy says)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty, Fact, Source, Vocab

    policy = policy or policies.current()
    s = sources(path)
    contract = Contract(boundary, consumer, ("Vocab",))
    decisions = []
    if tokenizer_size is None:
        decisions.append(load.cannot_check(boundary, consumer, "Vocab", f"{where}: the tokenizer's size was not read",
                                           policy))
    elif not s.candidates and s.rows is None:
        decisions.append(load.cannot_check(boundary, consumer, "Vocab",
                                           f"{path}: no tokenizer file, config vocab_size or embedding to compare with",
                                           policy))
    else:
        held = Fact("Vocab", Vocab(size=int(tokenizer_size),
                                   added=None if tokenizer_len is None else max(0, int(tokenizer_len) - int(tokenizer_size))),
                    Source("engine", f"{where}: the tokenizer the engine holds"), Certainty.VERIFIED)
        sizes = sorted({c[0] for c in s.candidates})
        model = s.rows
        rows_fact = None if model is None else Fact(
            "Vocab", Vocab(size=int(model)), Source("data" if s.rows_verified else "config", s.rows_where),
            Certainty.VERIFIED if s.rows_verified else Certainty.DECLARED)

        def broken(rule, declared, note):
            verdict, blocking = unrepaired(policy, "Vocab")
            return Decision(contract, "Vocab", verdict, RULES[rule], declared=declared, chosen=held, blocking=blocking,
                            note=note)

        if tokenizer_len is not None and model is not None and int(tokenizer_len) > int(model):
            decisions.append(broken("vocab_out_of_range", rows_fact,
                                    f"the tokenizer reaches id {int(tokenizer_len) - 1}; the model has {model} rows "
                                    f"({s.rows_where})"))
        elif len(sizes) > 1:
            named = ", ".join(f"{w} = {n}" for n, w in sorted(s.candidates))
            # the model's tokenizer is the source whose size IS the model's vocabulary (its rows or vocab_size);
            # anything short of that equality is a judgment, and a judgment is said as unknown (M15.3 review)
            mine = [c for c in s.candidates if model is not None and c[0] == model]
            if mine and int(tokenizer_size) == mine[0][0]:
                decisions.append(Decision(contract, "Vocab", Verdict.PASS, RULES["match"],
                                          declared=Fact("Vocab", Vocab(size=mine[0][0]), Source("file", mine[0][1]),
                                                        Certainty.DECLARED), chosen=held,
                                          note=f"the folder also carries another vocabulary ({named}); the engine "
                                               f"holds the model's"))
            elif mine and int(tokenizer_size) in sizes:
                decisions.append(broken("vocab_not_the_models",
                                        Fact("Vocab", Vocab(size=mine[0][0]), Source("file", mine[0][1]),
                                             Certainty.DECLARED),
                                        f"the folder declares {named}; the model's vocabulary is {model} "
                                        f"({s.rows_where}), so {mine[0][1]} is the model's tokenizer and the engine "
                                        f"built the other one"))
            else:
                decisions.append(load.cannot_check(boundary, consumer, "Vocab",
                                                   f"the folder declares {named}; the engine's tokenizer holds "
                                                   f"{tokenizer_size}, which is none of them, and the model's "
                                                   f"vocabulary ({model}) picks none", policy))
        elif sizes and int(tokenizer_size) not in sizes:
            decisions.append(load.cannot_check(boundary, consumer, "Vocab",
                                               f"the folder's tokenizer source says {sizes[0]} tokens "
                                               f"({s.candidates[0][1]}); the engine's tokenizer holds {tokenizer_size}: "
                                               f"which source it was built from cannot be told", policy))
        else:
            src = Fact("Vocab", Vocab(size=sizes[0]), Source("file", s.candidates[0][1]), Certainty.DECLARED) \
                if sizes else rows_fact
            decisions.append(Decision(contract, "Vocab", Verdict.PASS, RULES["match"], declared=src, chosen=held))
    if record:
        _tally.counts(boundary)["checks"] += 1
        if all(d.verdict is Verdict.PASS for d in decisions):
            _tally.passed(boundary, ["vocab"])
        load.enforce([d for d in decisions if d.verdict is not Verdict.PASS] or decisions, once_for=owner)
        if any(d.blocking for d in decisions):
            _tally.refused(boundary)
        elif any(d.verdict is Verdict.BROKEN for d in decisions):
            _tally.broken(boundary)
        _tally.tick(boundary)
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
