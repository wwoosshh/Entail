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
per folder. A folder whose declarations contradict each other - a legacy-export tokenizer.json (a normalizer that
prepends '\u2581' to every text) next to a tokenizer_config.json that declares legacy=false - is decided by where
the difference lies: a probe that differs only after a declared token, exactly as the flag's pipeline would have
it, is the folder's own disagreement (`sources_disagree`, unknown, the flag recorded as the conflicting source);
a probe that differs elsewhere, such as text that starts with whitespace, is `broken` as with any folder, because
every declaration the folder holds gives the same ids there. A difference the user's own build settings explain
(legacy=, add_prefix_space=, ... given to from_pretrained) is the user's choice (`user_choice`, unknown); the same
settings passed by an engine's or app's own code (SGLang retries a generic tokenizer with use_fast=False) are not
the user's, and the difference is broken with who set them (issue #37). No repair
is offered: a decision names the class the engine built and the first probe that differs, so the user can load the
declared tokenizer directly. Nothing here reads a device; an error inside entail never breaks the engine (the
adapters run this under load.safely).

Cost (M18.1, 38 popular folders on transformers 5.17): the first process on a folder builds the reference and pays
+241 ms at the median (+894 ms at most, a 33 MB tokenizer.json) on a tokenizer build of 189 ms; its probe ids are
kept in entail_logs/tokenizer_ids.json (per start folder) keyed by the files' stamp, the library version and
REFERENCE_VERSION, so the next process (vLLM and SGLang build the tokenizer three or four times per run) encodes
the probes with the engine's tokenizer and looks the added tokens up: +2 ms at the median, +36 ms at the 90th
percentile, within what one off/on pair per folder can resolve.
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
REFERENCE_VERSION = 2       # how the reference is built; bumped when that changes, so cached ids are not reused
SPM_FILES = ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model")
WATCHED = ("tokenizer.json", "tokenizer_config.json", "added_tokens.json", "tiktoken.model") + SPM_FILES
CACHE_NAME = "tokenizer_ids.json"
USER_KWARGS = ("legacy", "add_prefix_space", "split_special_tokens", "from_slow", "fix_mistral_regex", "use_fast")
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


def has_declaration(path) -> bool:
    """Whether the folder holds a tokenizer source at all (tokenizer.json, a sentencepiece model, tiktoken.model,
    vocab.json, vocab.txt, merges.txt; a source the check cannot run is said as unknown). A folder with
    tokenizer_config.json alone gives transformers a degenerate one-token tokenizer, about which the vocabulary
    check already says unknown: nothing is compared there (M18.1 review, finding 12a)."""
    return any(os.path.isfile(os.path.join(path, n))
               for n in ("tokenizer.json", "tiktoken.model", "vocab.json", "vocab.txt", "merges.txt") + SPM_FILES)


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


def declared_flags(path) -> dict:
    """The tokenizer_config.json flags that name a pipeline property, as declared (True or False): legacy
    (transformers' Llama classes: whether a segment after a special token gets a prepended space) and
    add_prefix_space (whether the text gets one at its start)."""
    p = os.path.join(path, "tokenizer_config.json")
    if not os.path.isfile(p):
        return {}
    try:
        cfg = json.load(open(p, encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return {k: cfg[k] for k in ("legacy", "add_prefix_space") if isinstance(cfg.get(k), bool)}


def special_strings(path) -> List[str]:
    """The strings the folder declares as tokens: the added tokens and the bos/eos/unk/pad of tokenizer_config.json,
    longest first. A probe text that contains one has a segment that follows a token."""
    out = set(declared_added(path)[0].values())
    p = os.path.join(path, "tokenizer_config.json")
    if os.path.isfile(p):
        try:
            cfg = json.load(open(p, encoding="utf-8"))
            for k in ("bos_token", "eos_token", "unk_token", "pad_token"):
                v = cfg.get(k)
                v = v.get("content") if isinstance(v, dict) else v
                if isinstance(v, str) and v:
                    out.add(v)
        except (ValueError, OSError):
            pass
    return sorted(out, key=len, reverse=True)


def flag_variant(path, flags: dict):
    """tokenizer.json's pipeline rebuilt the way transformers 5's Llama class does under the declared flag, when
    the flag and the file disagree: the Prepend normalizer dropped and Metaspace(prepend_scheme="first") for
    legacy=false, "never" for add_prefix_space=false; Metaspace("always") added for add_prefix_space=true or
    legacy=true over a file that prepends nothing. None when the declarations agree (the file already does what
    the flag says), or the file cannot be rebuilt. Returns (text -> ids, the scheme)."""
    p = os.path.join(path, "tokenizer.json")
    if not flags or not os.path.isfile(p):
        return None
    try:
        from tokenizers import Tokenizer

        d = json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    norm = d.get("normalizer")
    parts = norm.get("normalizers", [norm]) if isinstance(norm, dict) else []
    prepends = any(isinstance(n, dict) and n.get("type") == "Prepend" for n in parts)
    pre = d.get("pre_tokenizer")
    # the flags speak of the '▁' prefix of a sentencepiece-style pipeline (no pre-tokenizer, or Metaspace, with
    # spaces replaced by '▁'); a byte-level file (a Split or ByteLevel pre-tokenizer) declares another pipeline
    # altogether, which a class that rebuilds it with Metaspace drops whole - nothing there is the flag's doing
    # (the static corpus: DeepSeek-R1-0528-Qwen3-8B declares LlamaTokenizerFast over a byte-level tokenizer.json)
    metaspace_style = (pre is None or (isinstance(pre, dict) and pre.get("type") == "Metaspace")) and (
        prepends or any(isinstance(n, dict) and n.get("type") in ("Replace", "Prepend") for n in parts))
    if not metaspace_style:
        return None
    if flags.get("add_prefix_space") is False and prepends:
        scheme = "never"
    elif flags.get("legacy") is False and prepends:
        scheme = "first"
    elif (flags.get("add_prefix_space") is True or flags.get("legacy") is True) and not prepends:
        scheme = "always"
    else:
        return None
    kept = [n for n in parts if not (isinstance(n, dict) and n.get("type") == "Prepend")]
    d["normalizer"] = None if not kept else (kept[0] if len(kept) == 1 else {"type": "Sequence", "normalizers": kept})
    d["pre_tokenizer"] = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": False}
    try:
        t = Tokenizer.from_str(json.dumps(d))
        t.no_padding()
        t.no_truncation()
    except Exception:  # noqa: BLE001
        return None
    return (lambda s: list(t.encode(s, add_special_tokens=False).ids)), scheme


def excused_by_flag(path, got: List[List[int]], diff: List[int]) -> Tuple[List[int], str]:
    """Which of the differing probes the folder's own declared flag explains: tokenizer_config.json declares a
    prefix rule that contradicts tokenizer.json, and the probe encodes exactly as the flag's pipeline would. The
    legacy flag's documented meaning concerns text after a special token, so under it only a probe that contains
    a declared token can be excused; add_prefix_space concerns the start of the text, so any probe can (M18.1
    review, findings 2, 3 and 6). Returns (excused probe indexes, the flag as declared)."""
    flags = declared_flags(path)
    if not flags:
        return [], ""
    built = flag_variant(path, flags)
    if built is None:
        return [], ""
    variant, scheme = built
    specials = special_strings(path) if scheme == "first" else None
    out = []
    for i in diff:
        if specials is not None and not any(s in PROBES[i] for s in specials):
            continue
        try:
            if variant(PROBES[i]) == got[i]:
                out.append(i)
        except Exception:  # noqa: BLE001
            continue
    named = [k for k in ("add_prefix_space", "legacy") if k in flags and (
        (scheme == "never" and k == "add_prefix_space") or (scheme == "first" and k == "legacy")
        or (scheme == "always" and flags[k] is True))]
    return out, ", ".join(f"{k}={str(flags[k]).lower()}" for k in named)


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
        # the file can carry call-time settings (padding, truncation) that an engine's encode() removes, and a
        # training-time one (BPE dropout) that makes every call differ: the first two are cleared, the third is not
        # compared (M18.1 review, finding 5)
        try:
            t.no_padding()
            t.no_truncation()
        except Exception:  # noqa: BLE001
            pass
        if getattr(getattr(t, "model", None), "dropout", None):
            return None, "tokenizer.json", ["tokenizer.json sets BPE dropout (a training-time setting): the probe "
                                            "texts are not compared"]
        return (lambda s: list(t.encode(s, add_special_tokens=False).ids),
                f"tokenizer.json (tokenizers {tokenizers.__version__})", [])
    for name in SPM_FILES:
        p = os.path.join(path, name)
        if os.path.isfile(p):
            # sentencepiece encodes a special token's text as pieces where every transformers tokenizer matches the
            # token; a faithful reference must split the text on the declared special tokens first, and that path
            # has not been measured on sentencepiece-only folders (M18.1 review, finding 1): not compared
            return None, name, [f"{name}: a sentencepiece reference is not compared until it is measured on "
                                f"sentencepiece-only folders (the added tokens are)"]
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


def _library() -> str:
    """The library and version the reference is built with, part of the cache key: ids built by another version
    are not reused (M18.1 review, finding 10)."""
    try:
        import tokenizers

        return f"tokenizers {tokenizers.__version__}"
    except ImportError:
        return "no tokenizers"


def _from_file_cache(path, stamp) -> Optional[Reference]:
    global _FILE_CACHE
    p = _cache_path()
    if p is None:
        return None
    if _FILE_CACHE is None:
        try:
            _FILE_CACHE = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
        except Exception:  # noqa: BLE001
            _FILE_CACHE = {}
    e = _FILE_CACHE.get(os.path.abspath(path))
    if not e or [tuple(x) for x in e.get("stamp", [])] != list(stamp) or e.get("probes") != PROBES_DIGEST \
            or e.get("reference") != REFERENCE_VERSION or e.get("library") != _library():
        return None
    return Reference(where=e.get("where", ""), ids=e.get("ids"),
                     added={int(k): v for k, v in (e.get("added") or {}).items()}, added_where=e.get("added_where", ""),
                     problems=list(e.get("problems", [])))


def _to_file_cache(path, stamp, r: Reference) -> None:
    """Keep the reference's probe ids per start folder (entail_logs/tokenizer_ids.json). A failure that depends on
    the environment (a library not installed) is not kept: another environment may run the declaration."""
    global _FILE_CACHE
    p = _cache_path()
    if p is None or any("not installed" in x for x in r.problems):
        return
    try:
        import threading

        if _FILE_CACHE is None:
            _FILE_CACHE = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
        _FILE_CACHE[os.path.abspath(path)] = {"stamp": [list(x) for x in stamp], "probes": PROBES_DIGEST,
                                              "reference": REFERENCE_VERSION, "library": _library(),
                                              "where": r.where, "ids": r.ids,
                                              "added": {str(k): v for k, v in r.added.items()},
                                              "added_where": r.added_where, "problems": r.problems}
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(_FILE_CACHE), f)
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001 - a cache that cannot be written is only a cost, never an error
        pass


def reference(path) -> Reference:
    """The declared tokenizer of the folder, run on PROBES; once per folder per process and once per start folder
    (the machine cache lives in the start folder's entail_logs)."""
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
    try:
        ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)   # no length warning consumed
    except TypeError:
        ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "ids"):            # a tokenizers.Encoding (a stand-in that is the library itself)
        ids = ids.ids
    return [int(i) for i in ids]


def _engine_probe_ids(tokenizer) -> List[List[int]]:
    """The engine's ids for PROBES. transformers' encode() resets the backend's padding and truncation as a side
    effect; they are put back afterwards (M18.1 review, finding 13)."""
    bt = getattr(tokenizer, "backend_tokenizer", None)
    padding = truncation = None
    if bt is not None:
        try:
            padding, truncation = bt.padding, bt.truncation
        except Exception:  # noqa: BLE001
            bt = None
    try:
        return [_engine_ids(tokenizer, s) for s in PROBES]
    finally:
        if bt is not None:
            try:
                if padding:
                    bt.enable_padding(**padding)
                if truncation:
                    bt.enable_truncation(**truncation)
            except Exception:  # noqa: BLE001 - a backend whose settings cannot be put back keeps encode()'s reset
                pass


def _short(ids: List[int], n: int = 12) -> str:
    return str(ids[:n])[:-1] + (", ...]" if len(ids) > n else "]")


def check(boundary: str, consumer: str, path: str, tokenizer, where: str, policy=None, owner=None,
          record: bool = True, user_kwargs: Optional[dict] = None, set_by: Optional[str] = None) -> list:
    """Decide the tokenizer the engine built (`tokenizer`: anything with encode(text, add_special_tokens=False) and
    convert_tokens_to_ids) against the folder's declared tokenizer, run. Returns the decisions; with `record` they
    are also recorded through load.enforce (the run goes on, or stops where the policy says). `user_kwargs`: the
    build settings given to from_pretrained (legacy, add_prefix_space, ...): a difference they explain is the
    user's choice, not the engine's fault - unless `set_by` names the installed package whose code gave them."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty, Fact, Source, Tokenization

    policy = policy or policies.current()
    r = reference(path)
    contract = Contract(boundary, consumer, ("Tokenization",))
    chosen_by_user = {k: v for k, v in (user_kwargs or {}).items() if k in USER_KWARGS and v is not None}
    decisions = []

    def mismatch(rule, declared, held, note):
        """A difference: the user's own build settings explain it (unknown, user_choice), else broken - with the
        settings and who set them, when an engine's or app's code did."""
        asked = ", ".join(f"{k}={v!r}" for k, v in sorted(chosen_by_user.items()))
        if chosen_by_user and not set_by:
            return Decision(contract, "Tokenization", Verdict.UNKNOWN, RULES["user_choice"], declared=declared,
                            chosen=held, note=f"{note}; the tokenizer was built with {asked} given by the user, "
                                              f"which is the user's choice against the file")
        if chosen_by_user:
            note = f"{note}; the tokenizer was built with {asked}, set by {set_by}, not by the user"
        verdict, blocking = unrepaired(policy, "Tokenization")
        return Decision(contract, "Tokenization", verdict, RULES[rule], declared=declared, chosen=held,
                        blocking=blocking, note=note)

    # --- the declared added tokens: content -> id, a dict lookup each (all of them: M18.1 review, finding 8) ---
    n_added = None
    lookup = getattr(tokenizer, "convert_tokens_to_ids", None)
    if r.added and not callable(lookup):
        decisions.append(load.cannot_check(boundary, consumer, "Tokenization",
                                           f"{where} has no convert_tokens_to_ids: the {len(r.added)} declared added "
                                           f"tokens are not compared", policy))
    elif r.added:
        items = sorted(r.added.items())
        n_added = len(items)
        wrong, errors = [], 0
        unk = getattr(tokenizer, "unk_token_id", None)
        for tid, content in items:
            try:
                got = lookup(content)
            except Exception:  # noqa: BLE001 - a class that raises for a token it lacks: the token is missing
                got, errors = None, errors + 1
            if isinstance(got, list):
                got = got[0] if len(got) == 1 else None
            if got != tid:
                wrong.append((content, tid, got, got is None or (unk is not None and got == unk)))
        if wrong:
            content, tid, got, missing = wrong[0]
            what = "not a token of the engine's tokenizer" if missing else f"id {got}"
            declared = Fact("Tokenization", Tokenization(digest=digest([[tid]]), probes=0, added=n_added),
                            Source("file", r.added_where), Certainty.DECLARED)
            held = Fact("Tokenization", Tokenization(digest=digest([[got if isinstance(got, int) else -1]]),
                                                     probes=0, added=n_added), Source("engine", where),
                        Certainty.VERIFIED)
            note = f"{r.added_where} declares {content!r} as id {tid}; {where} gives it {what}"
            if len(wrong) > 1:
                note += f"; {len(wrong)} of {n_added} declared added tokens differ"
            if errors:
                note += f" ({errors} raised on lookup)"
            decisions.append(mismatch("added_token_id", declared, held, note))
    # --- the probe texts ---
    if r.ids is None:
        why = "; ".join(r.problems) if r.problems else "the declared tokenizer could not be run"
        if n_added and not decisions:
            why += f" ({n_added} declared added tokens compared: all match)"
        decisions.append(load.cannot_check(boundary, consumer, "Tokenization", f"{where}: {why}", policy))
    else:
        try:
            got = _engine_probe_ids(tokenizer)
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
            excused, flag = (excused_by_flag(path, got, diff) if diff else ([], ""))
            remaining = [i for i in diff if i not in excused]
            if remaining:
                i = remaining[0]
                note = (f"{len(remaining)} of {len(PROBES)} probe texts encode differently; first: {PROBES[i]!r:.60}: "
                        f"{where} gives {_short(got[i])}, {r.where} gives {_short(r.ids[i])}")
                if excused:
                    note += (f"; {len(excused)} further differ only after a declared token, as tokenizer_config.json's "
                             f"{flag} would have it")
                decisions.append(mismatch("tokenizer_ids", declared, held, note))
            elif diff:
                # every difference lies after a declared token and is what the folder's own flag asks for: the
                # folder's declarations disagree (a legacy-export tokenizer.json next to legacy=false), and which
                # one the model was trained with is not decidable here - said with both, as the sources rule
                i = diff[0]
                conflict = Fact("Tokenization", Tokenization(digest=digest(got), probes=len(PROBES), added=n_added),
                                Source("file", f"tokenizer_config.json {flag} (the flag's pipeline gives the "
                                               f"engine's ids)"), Certainty.DECLARED)
                setting = policy.unknown_setting("Tokenization", True)
                blocking = policy.mode == "debug" or setting in ("require", "stop") or \
                    policy.on_source_conflict == "stop"
                decisions.append(Decision(
                    Contract(boundary, consumer, ("Tokenization",), ("Tokenization",)), "Tokenization",
                    Verdict.REFUSED if blocking else Verdict.UNKNOWN, RULES["sources_disagree"], declared=declared,
                    chosen=held, blocking=blocking, conflict=(conflict,),
                    note=f"the folder's declarations disagree on the prefix space: tokenizer.json's pipeline and "
                         f"tokenizer_config.json's {flag} say different things, and the {len(diff)} differing probe "
                         f"texts encode exactly as the flag's pipeline would have it (first: {PROBES[i]!r:.60}: "
                         f"{where} gives {_short(got[i])}, tokenizer.json gives {_short(r.ids[i])}); which one the "
                         f"model was trained with cannot be told here"))
            elif not any(d.verdict in (Verdict.BROKEN, Verdict.REFUSED) for d in decisions):
                note = f"{len(PROBES)} probe texts encode alike"
                if n_added:
                    note += f"; {n_added} declared added tokens match"
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
