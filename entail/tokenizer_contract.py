"""tokenizer_contract: the tokenizer an engine built against the tokenizer the model folder declares, compared by
what they do (LIBRARY_DESIGN.md 11, M18; ROADMAP M18.1).

The vocabulary check (vocab_contract) compares sizes: it tells a tokenizer built from the wrong file. It does not
tell a tokenizer built from the right file by the wrong class. transformers 5.10.2 built deepseek-coder's tokenizer
as LlamaTokenizer and 4 of 4 strings encoded to other ids than the folder's tokenizer.json gives
(transformers#46489); 5.8.0 built Granite's as GPT2Tokenizer and lost its pre-tokenizer (#45812); 5.4.0 converted
Kimi-K2.5's tiktoken tokenizer and gave `</think>` the id of `<|media_end|>` (#45356); 5.9 mapped
DeepSeek-R1-Distill's declared class to another whose decode carries byte-level artefacts (#46710). Every one of
them passed the size check: the count was right and the ids were wrong.

The declaration is the folder's tokenizer file, and it can be run: the `tokenizers` library reads tokenizer.json,
`sentencepiece` reads a sentencepiece model, and tokenizer_config.json's added_tokens_decoder (else tokenizer.json's
added_tokens, else added_tokens.json) names every added token with its id. The reference built from the
declaration and the tokenizer the engine built encode the same fixed probe texts (PROBES), and the ids must be the
same - exactly, no tolerance. The rules are here, once:

  tokenizer_ids    the engine's tokenizer encodes a probe text to other ids than the folder's declared tokenizer
  added_token_id   the engine's tokenizer gives a declared added token (content -> id) another id

A declaration that cannot be run - no tokenizer file, its library not installed, a tiktoken.model (which declares
the ranks and not the pre-tokenization pattern, so only the added tokens are compared) - is said as unknown, once
per folder. So is a folder whose declarations contradict each other (a legacy-mode tokenizer.json next to a
tokenizer_config.json that declares legacy=false: the engine follows the flag and the ids differ from the file on a
text that starts with whitespace; which one the model was trained with is not decidable here). No repair is offered: a decision names the class the engine built and the first probe that differs, so
the user can load the declared tokenizer directly. Nothing here reads a device; an error inside entail never breaks
the engine (the adapters run this under load.safely).

Cost: the reference is built once per folder per process (`tokenizers` reads a 7 MB tokenizer.json in tens of
milliseconds) and its probe ids are kept in entail_logs/tokenizer_ids.json keyed by the files' stamp, so the next
process (vLLM and SGLang build the tokenizer three or four times per run) only encodes the probes with the engine's
tokenizer: a few milliseconds.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import tally as _tally

RULE_NAMES = ("tokenizer_ids", "added_token_id")

# Fixed probe texts: a plain sentence, code with indentation and newlines, leading and trailing spaces, digits and
# punctuation, several scripts, emoji and symbols, tabs and Windows line ends, markup that is text unless the folder
# declares it as a token, and a long run of one character (merges). They are part of the rule: changing them changes
# the digest, so the file cache is keyed by their digest too.
PROBES = (
    "How are you doing?",
    "The quick brown fox jumps over the lazy dog.",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "   leading spaces and trailing   ",
    "Hello, world! 1234 56.78 -9 (a+b)*c",
    "안녕하세요, 세계. 東京タワー 北京 مرحبا Привет",
    "emoji \U0001f642\U0001f680 and symbols → ∑ ≤ €",
    "tabs\tand\r\nwindows newlines",
    "<|endoftext|> <s> </s> [INST] <|im_start|>",
    "a" * 300,
)
PROBES_DIGEST = hashlib.sha256("\x1f".join(PROBES).encode("utf-8")).hexdigest()[:16]
SPM_FILES = ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model")
WATCHED = ("tokenizer.json", "tokenizer_config.json", "added_tokens.json", "tiktoken.model") + SPM_FILES
CACHE_NAME = "tokenizer_ids.json"
ADDED_LIMIT = 4096          # declared added tokens compared per folder (a dict lookup each; Gemma declares ~100)
_CACHE: dict = {}           # folder -> (stamp, Reference)
_FILE_CACHE = None


@dataclass
class Reference:
    """The declared tokenizer, run: the ids it gives PROBES (None when the declaration cannot be run), and the
    added tokens the folder declares (id -> content)."""
    where: str = ""
    ids: Optional[List[List[int]]] = None
    added: Dict[int, str] = field(default_factory=dict)
    added_where: str = ""
    problems: List[str] = field(default_factory=list)


def digest(ids: List[List[int]]) -> str:
    return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()[:16]


def _stamp(path) -> tuple:
    out = []
    for name in WATCHED:
        try:
            st = os.stat(os.path.join(path, name))
            out.append((name, st.st_size, int(st.st_mtime)))
        except OSError:
            continue
    return tuple(out)


def declared_added(path) -> Tuple[Dict[int, str], str]:
    """The added tokens the folder declares, id -> content, and where from: tokenizer_config.json's
    added_tokens_decoder (what transformers itself reads), else tokenizer.json's added_tokens, else added_tokens.json."""
    p = os.path.join(path, "tokenizer_config.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p, encoding="utf-8")).get("added_tokens_decoder")
            if isinstance(d, dict) and d:
                out = {}
                for k, v in d.items():
                    if isinstance(v, dict) and isinstance(v.get("content"), str) and str(k).isdigit():
                        out[int(k)] = v["content"]
                if out:
                    return out, "tokenizer_config.json added_tokens_decoder"
        except (ValueError, OSError):
            pass
    p = os.path.join(path, "tokenizer.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p, encoding="utf-8")).get("added_tokens")
            if isinstance(d, list) and d:
                out = {int(t["id"]): t["content"] for t in d
                       if isinstance(t, dict) and isinstance(t.get("id"), int) and isinstance(t.get("content"), str)}
                if out:
                    return out, "tokenizer.json added_tokens"
        except (ValueError, OSError, TypeError):
            pass
    p = os.path.join(path, "added_tokens.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            if isinstance(d, dict) and d:
                out = {int(v): k for k, v in d.items() if isinstance(k, str) and isinstance(v, int)}
                if out:
                    return out, "added_tokens.json"
        except (ValueError, OSError):
            pass
    return {}, ""


def declarations_disagree(path) -> Optional[str]:
    """When the folder's own declarations contradict each other on how a text begins, nothing here can say which
    one the model was trained with (M18.1 E2: TinyLlama, tiny-random-Llama). A Llama-2-era tokenizer.json exported
    in transformers' legacy mode carries a normalizer that prepends '▁' to every text, while its
    tokenizer_config.json declares legacy=false (or add_prefix_space=false); transformers 4 took the file as it was,
    transformers 5's LlamaTokenizer follows the flag, and the ids differ on a text that starts with whitespace or
    follows a special token. Returns the explanation, or None when the declarations agree (or say nothing)."""
    p = os.path.join(path, "tokenizer_config.json")
    if not os.path.isfile(p):
        return None
    try:
        cfg = json.load(open(p, encoding="utf-8"))
    except (ValueError, OSError):
        return None
    flags = [k for k in ("legacy", "add_prefix_space") if cfg.get(k) is False]
    if not flags:
        return None
    p = os.path.join(path, "tokenizer.json")
    if not os.path.isfile(p):
        return None
    try:
        norm = json.load(open(p, encoding="utf-8")).get("normalizer")
    except (ValueError, OSError):
        return None
    parts = norm.get("normalizers", [norm]) if isinstance(norm, dict) else []
    if not any(isinstance(n, dict) and n.get("type") == "Prepend" for n in parts):
        return None
    return (f"the folder's declarations disagree: tokenizer.json prepends '▁' to every text (a legacy export: "
            f"normalizer Prepend) while tokenizer_config.json declares {', '.join(f'{k}=false' for k in flags)}; "
            f"the engine follows the flag")


def _encoder(path) -> Tuple[Optional[Callable[[str], List[int]]], str, List[str]]:
    """(encode, where, problems): the declared tokenizer as a function text -> ids without special tokens, or None
    with the reason it cannot be run."""
    p = os.path.join(path, "tokenizer.json")
    if os.path.isfile(p):
        try:
            import tokenizers
            from tokenizers import Tokenizer
        except ImportError:
            return None, "tokenizer.json", ["the tokenizers library is not installed: tokenizer.json cannot be run"]
        try:
            t = Tokenizer.from_file(p)
        except Exception as e:  # noqa: BLE001 - a file this tokenizers version does not read
            return None, "tokenizer.json", [f"tokenizers {tokenizers.__version__} cannot read tokenizer.json: "
                                            f"{type(e).__name__}: {str(e)[:120]}"]
        return (lambda s: list(t.encode(s, add_special_tokens=False).ids),
                f"tokenizer.json (tokenizers {tokenizers.__version__})", [])
    for name in SPM_FILES:
        p = os.path.join(path, name)
        if os.path.isfile(p):
            try:
                import sentencepiece as spm
            except ImportError:
                return None, name, [f"the sentencepiece library is not installed: {name} cannot be run"]
            try:
                sp = spm.SentencePieceProcessor(model_file=p)
            except Exception as e:  # noqa: BLE001
                return None, name, [f"sentencepiece cannot read {name}: {type(e).__name__}: {str(e)[:120]}"]
            return (lambda s: [int(i) for i in sp.encode(s, out_type=int)],
                    f"{name} (sentencepiece {getattr(spm, '__version__', '?')})", [])
    if os.path.isfile(os.path.join(path, "tiktoken.model")):
        return None, "tiktoken.model", ["tiktoken.model declares the ranks and not the pre-tokenization pattern: "
                                        "the probe texts are not compared, the added tokens are"]
    return None, "", ["no tokenizer file the declaration can be run from (tokenizer.json, a sentencepiece model)"]


def _cache_path() -> Optional[str]:
    try:
        from . import record

        folder = record.log_dir()
    except Exception:  # noqa: BLE001
        return None
    return os.path.join(folder, CACHE_NAME) if folder else None


def _from_file_cache(path, stamp) -> Optional[Reference]:
    global _FILE_CACHE
    p = _cache_path()
    if p is None:
        return None
    if _FILE_CACHE is None:
        try:
            _FILE_CACHE = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
        except (ValueError, OSError):
            _FILE_CACHE = {}
    e = _FILE_CACHE.get(os.path.abspath(path))
    if not e or [tuple(x) for x in e.get("stamp", [])] != list(stamp) or e.get("probes") != PROBES_DIGEST:
        return None
    return Reference(where=e.get("where", ""), ids=e.get("ids"),
                     added={int(k): v for k, v in (e.get("added") or {}).items()}, added_where=e.get("added_where", ""),
                     problems=list(e.get("problems", [])))


def _to_file_cache(path, stamp, r: Reference) -> None:
    global _FILE_CACHE
    p = _cache_path()
    if p is None:
        return
    try:
        if _FILE_CACHE is None:
            _FILE_CACHE = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
        _FILE_CACHE[os.path.abspath(path)] = {"stamp": [list(x) for x in stamp], "probes": PROBES_DIGEST,
                                              "where": r.where, "ids": r.ids,
                                              "added": {str(k): v for k, v in r.added.items()},
                                              "added_where": r.added_where, "problems": r.problems}
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_FILE_CACHE, f)
        os.replace(tmp, p)
    except (ValueError, OSError):   # a cache that cannot be written is only a cost, never an error
        pass


def reference(path) -> Reference:
    """The declared tokenizer of the folder, run on PROBES; once per folder per process and once per machine."""
    stamp = _stamp(path)
    hit = _CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    r = _from_file_cache(path, stamp)
    if r is None:
        encode, where, problems = _encoder(path)
        r = Reference(where=where, problems=problems)
        if encode is not None:
            try:
                r.ids = [encode(s) for s in PROBES]
            except Exception as e:  # noqa: BLE001 - the declared tokenizer fails on a probe: not run
                r.ids, r.problems = None, [f"{where} failed on a probe text: {type(e).__name__}: {str(e)[:120]}"]
        r.added, r.added_where = declared_added(path)
        _to_file_cache(path, stamp, r)
    _CACHE[path] = (stamp, r)
    return r


def _engine_ids(tokenizer, text) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "ids"):            # a tokenizers.Encoding (a stand-in that is the library itself)
        ids = ids.ids
    return [int(i) for i in ids]


def _short(ids: List[int], n: int = 12) -> str:
    return str(ids[:n])[:-1] + (", ...]" if len(ids) > n else "]")


def check(boundary: str, consumer: str, path: str, tokenizer, where: str, policy=None, owner=None,
          record: bool = True) -> list:
    """Decide the tokenizer the engine built (`tokenizer`: anything with encode(text, add_special_tokens=False) and
    convert_tokens_to_ids) against the folder's declared tokenizer, run. Returns the decisions; with `record` they
    are also recorded through load.enforce (the run goes on, or stops where the policy says)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty, Fact, Source, Tokenization

    policy = policy or policies.current()
    r = reference(path)
    contract = Contract(boundary, consumer, ("Tokenization",))
    decisions = []
    n_added = None
    # --- the declared added tokens: content -> id, a dict lookup each ---
    added_note = ""
    if r.added:
        items = sorted(r.added.items())[:ADDED_LIMIT]
        n_added = len(items)
        wrong = []
        unk = getattr(tokenizer, "unk_token_id", None)
        for tid, content in items:
            try:
                got = tokenizer.convert_tokens_to_ids(content)
            except Exception as e:  # noqa: BLE001 - a class without the table: cannot be compared
                wrong, added_note = [], f"the added tokens could not be looked up ({type(e).__name__})"
                break
            if isinstance(got, list):
                got = got[0] if len(got) == 1 else None
            if got != tid:
                wrong.append((content, tid, got, got is None or (unk is not None and got == unk)))
        if wrong:
            content, tid, got, missing = wrong[0]
            verdict, blocking = unrepaired(policy, "Tokenization")
            what = "not a token of the engine's tokenizer" if missing else f"id {got}"
            decisions.append(Decision(
                contract, "Tokenization", verdict, RULES["added_token_id"],
                declared=Fact("Tokenization", Tokenization(digest=digest([[tid]]), probes=0, added=n_added),
                              Source("file", r.added_where), Certainty.DECLARED),
                chosen=Fact("Tokenization", Tokenization(digest=digest([[got if isinstance(got, int) else -1]]),
                                                         probes=0, added=n_added),
                            Source("engine", where), Certainty.VERIFIED),
                blocking=blocking,
                note=f"{r.added_where} declares {content!r} as id {tid}; {where} gives it {what}"
                     + (f"; {len(wrong)} of {n_added} declared added tokens differ" if len(wrong) > 1 else "")))
    # --- the probe texts ---
    if r.ids is None:
        why = "; ".join(r.problems) if r.problems else "the declared tokenizer could not be run"
        if r.added and not decisions:
            why += f" ({n_added} declared added tokens compared: all match)"
        decisions.append(load.cannot_check(boundary, consumer, "Tokenization", f"{where}: {why}", policy))
    else:
        try:
            got = [_engine_ids(tokenizer, s) for s in PROBES]
        except Exception as e:  # noqa: BLE001 - an engine tokenizer that cannot encode text
            got = None
            decisions.append(load.cannot_check(boundary, consumer, "Tokenization",
                                               f"{where} could not encode the probe texts: {type(e).__name__}: "
                                               f"{str(e)[:120]}", policy))
        if got is not None:
            declared = Fact("Tokenization", Tokenization(digest=digest(r.ids), probes=len(PROBES), added=n_added),
                            Source("file", r.where), Certainty.DECLARED)
            held = Fact("Tokenization", Tokenization(digest=digest(got), probes=len(PROBES), added=n_added),
                        Source("engine", where), Certainty.VERIFIED)
            diff = [i for i in range(len(PROBES)) if got[i] != r.ids[i]]
            if diff:
                i = diff[0]
                what = (f"{len(diff)} of {len(PROBES)} probe texts encode differently; first: {PROBES[i]!r:.60}: "
                        f"{where} gives {_short(got[i])}, {r.where} gives {_short(r.ids[i])}")
                disagree = declarations_disagree(path)
                if disagree:
                    # two declarations, one followed: which the model was trained with is not decidable here
                    decisions.append(load.cannot_check(boundary, consumer, "Tokenization",
                                                       f"{disagree}; {what}; which one the model was trained "
                                                       f"with cannot be told here", policy))
                else:
                    verdict, blocking = unrepaired(policy, "Tokenization")
                    decisions.append(Decision(contract, "Tokenization", verdict, RULES["tokenizer_ids"],
                                              declared=declared, chosen=held, blocking=blocking, note=what))
            elif not decisions:
                note = f"{len(PROBES)} probe texts encode alike"
                if n_added:
                    note += f"; {n_added} declared added tokens match"
                if added_note:
                    note += f"; {added_note}"
                decisions.append(Decision(contract, "Tokenization", Verdict.PASS, RULES["match"], declared=declared,
                                          chosen=held, note=note))
    if record:
        _tally.counts(boundary)["checks"] += 1
        if all(d.verdict is Verdict.PASS for d in decisions):
            _tally.passed(boundary, ["tokenization"])
        if any(d.blocking for d in decisions):    # counted before enforce, which raises where the policy stops
            _tally.refused(boundary)
        elif any(d.verdict is Verdict.BROKEN for d in decisions):
            _tally.broken(boundary)
        _tally.tick(boundary)
        load.enforce([d for d in decisions if d.verdict is not Verdict.PASS] or decisions, once_for=owner)
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: Optional[str] = None) -> None:
    _CACHE.clear()
    if boundary:
        _tally.reset(boundary)
