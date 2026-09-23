"""readers: turn what artifacts already declare into facts (LIBRARY_DESIGN.md 4.2; ROADMAP M2.1).

One reader per kind of artifact:
  hf_config    config.json of a Hugging Face model: ModelProps, Rotary, Layout (quantization_config)
  hf_template  tokenizer_config.json and chat_template.jinja: Template (the chat template's hash)
  diffusers    scheduler/scheduler_config.json and vae/config.json: Prediction, LatentScale
  safetensors  a .safetensors header (metadata and marker keys, never the tensors): Prediction
  gguf         a .gguf header: ModelProps, Rotary, Template

Rules every reader follows:
  - A fact is emitted only for what the artifact states; nothing is filled in from defaults.
  - Every fact names the file and the keys it was read from (Source.where), so the ledger can say where a value
    came from. Two statements of the same thing are two facts; `sources.merge` finds a disagreement.
  - A stated value that vocabulary v1 cannot represent (a rope type it does not know, per-layer RoPE, a
    quantization method it has no layout for) becomes a problem in the result, never a silent omission.
  - Key names and value spellings come from data/aliases.json.
"""
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from typing import List

from . import gguf
from .facts import VOCAB_VERSION, Certainty, Fact, LatentScale, Layout, ModelProps, Prediction, Rotary, Source, Template

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aliases.json"), encoding="utf-8") as _f:
    ALIASES = json.load(_f)
_VALUES = ALIASES["values"]


@dataclass
class ReadResult:
    facts: List[Fact] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    def extend(self, other):
        self.facts.extend(other.facts)
        self.problems.extend(other.problems)


def _emit(result, name, build, source_kind, where):
    """Build the value; a value outside the vocabulary becomes a problem instead of a fact."""
    try:
        value = build()
    except ValueError as e:
        result.problems.append(f"{where}: {e} (not representable in vocabulary v{VOCAB_VERSION})")
        return
    result.facts.append(Fact(name, value, Source(source_kind, where), Certainty.DECLARED))


def _first(d, names):
    """(key, value) of the first of `names` that `d` holds with a value other than None."""
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return n, d[n]
    return None, None


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prediction_kind(text):
    return _VALUES["prediction_kind"].get(str(text).strip().lower())


def flag(text):
    if isinstance(text, bool):
        return text
    return _VALUES["flag"].get(str(text).strip().lower())


def rotary_of(params):
    """A Rotary value from one RoPE parameters dict ({"rope_type": ..., "rope_theta": ..., "factor": ...}), with the
    problems of representing it: (Rotary or None, [problem, ...]). Used by adapters to say what a config holds."""
    r = ReadResult()
    rk = ALIASES["Rotary"]["hf_config"]
    _, theta = _first(params or {}, rk["theta"])
    _rotary(r, params, theta, None, "rope_parameters", "engine")
    return (r.facts[0].value if r.facts else None), r.problems


# --- Hugging Face config.json ----------------------------------------------------------------------------------

def _rotary(result, spec, theta, theta_key, where, source_kind="config"):
    """One Rotary fact from a scaling/parameters dict (may be None) and a base."""
    keys = ALIASES["Rotary"]["hf_config"]
    spec = spec or {}
    type_key, raw_type = _first(spec, keys["type"])
    rope_type = _VALUES["rope_type"].get(str(raw_type).lower()) if raw_type is not None else "default"
    if rope_type is None:
        result.problems.append(f"{where}: rope type {raw_type!r} is not in vocabulary v{VOCAB_VERSION}")
        return
    _, factor = _first(spec, keys["factor"])
    _, omp = _first(spec, keys["original_max_position"])
    carried = set(keys["type"] + keys["factor"] + keys["original_max_position"] + keys["theta"])
    left = sorted(k for k, v in spec.items() if k not in carried and v is not None)
    if left:   # e.g. llama3's high_freq_factor: stated, but v1 has no field for it; say so instead of dropping it
        result.problems.append(f"{where}: RoPE keys {left} are not in vocabulary v{VOCAB_VERSION}; the Rotary fact "
                               f"does not carry them")
    _emit(result, "Rotary", lambda: Rotary(rope_type, theta=theta, factor=factor, original_max_position=omp),
          source_kind, where)


def _props(result, props, used, source_kind, file):
    """ModelProps from the fields a file states. A field outside the vocabulary becomes a problem on its own, so a
    bad sliding_window does not take a good softcap down with it."""
    good, keys = {}, []
    for (fld, value), key in zip(props.items(), used):
        try:
            ModelProps(**{fld: value})
        except ValueError as e:
            result.problems.append(f"{file}#{key}: {e} (not representable in vocabulary v{VOCAB_VERSION})")
            continue
        good[fld] = value
        keys.append(key)
    if good:
        _emit(result, "ModelProps", lambda: ModelProps(**good), source_kind, f"{file}#{','.join(keys)}")


class HfConfig:
    name = "hf_config"

    def applies_to(self, path):
        return os.path.isfile(os.path.join(path, "config.json")) if os.path.isdir(path) else \
            os.path.basename(path) == "config.json"

    def read(self, path):
        file = os.path.join(path, "config.json") if os.path.isdir(path) else path
        return read_hf_dict(_load_json(file), file)


def read_hf_dict(cfg, label, source_kind="config", from_object=False):
    """The facts a Hugging Face config states, from its dict: config.json as read from disk, or the config object an
    engine holds (config_dict below, from_object=True). `label` names it in every Source."""
    file = label
    r = ReadResult()
    nested = isinstance(cfg.get("text_config"), dict)
    text, prefix = (cfg["text_config"], "text_config.") if nested else (cfg, "")

    # ModelProps: the model's own requirements of whatever runs its attention
    k = ALIASES["ModelProps"]["hf_config"]
    props, used = {}, []
    key, v = _first(text, k["softcap"])
    if key:
        props["softcap"], used = v, used + [prefix + key]
    key, v = _first(text, k["sliding_window"])
    _, enabled = _first(text, k["window_enabled"])
    if key and enabled is not False:       # a window the config switches off is not a requirement
        props["sliding_window"], used = v, used + [prefix + key]
    for scope, p in ((cfg, ""), (text, prefix)):
        key, v = _first(scope, k["tie"])
        if key:
            props["tie_word_embeddings"], used = v, used + [p + key]
            break
    _props(r, props, used, source_kind, file)

    # Rotary: transformers 5 writes rope_parameters; older files write rope_theta and rope_scaling
    rk = ALIASES["Rotary"]["hf_config"]
    pkey, params = _first(text, rk["parameters"])
    if pkey:
        if isinstance(params, dict) and params and all(isinstance(x, dict) for x in params.values()):
            r.problems.append(f"{file}#{prefix}{pkey}: RoPE set per layer type ({sorted(params)}) "
                              f"is not in vocabulary v{VOCAB_VERSION}")
        elif isinstance(params, dict):
            _, theta = _first(params, rk["theta"])
            _rotary(r, params, theta, None, f"{file}#{prefix}{pkey}", source_kind)
    tkey, theta = _first(text, rk["theta"])
    skey, scaling = _first(text, rk["scaling"])
    # config.json may state both spellings: two facts, and sources.merge finds a disagreement. In the config object an
    # engine holds, the model reads rope_parameters only; a leftover old-name attribute is not a declaration.
    if (tkey or skey) and not (pkey and from_object):
        where = f"{file}#" + ",".join(prefix + x for x in (tkey, skey) if x)
        _rotary(r, scaling if isinstance(scaling, dict) else None, theta, tkey, where, source_kind)

    # Layout of quantized weights
    qk = ALIASES["Layout"]["hf_quantization_config"]
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        where = f"{file}#quantization_config"
        _, method = _first(qc, qk["method"])
        if method == "fp8":
            _, block = _first(qc, qk["block"])
            _, fmt = _first(qc, qk["fmt"])
            _, scale = _first(qc, qk["scale_format"])
            if block:
                _emit(r, "Layout", lambda: Layout("fp8_block", dtype=_VALUES["fp8_format"].get(fmt),
                                                  block=tuple(block), scale_format=scale), source_kind, where)
            else:
                r.problems.append(f"{where}: per-tensor fp8 (no weight_block_size) is not in vocabulary v1")
        elif method in ("awq", "gptq"):
            _, bits = _first(qc, qk["bits"])
            _, gs = _first(qc, qk["group_size"])
            if bits == 4:
                block = (gs,) if isinstance(gs, int) and gs > 0 else None
                _emit(r, "Layout", lambda: Layout("int4_packed", block=block), source_kind, where)
            else:
                r.problems.append(f"{where}: {method} with {bits} bits is not in vocabulary v1")
        else:
            r.problems.append(f"{where}: quantization method {method!r} is not in vocabulary v1")
    return r


def config_dict(obj):
    """A plain dict of the config object an engine holds: PretrainedConfig.to_dict(), or its attributes (nested
    objects become dicts), so read_hf_dict reads it the way it reads config.json."""
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            if isinstance(d, dict):
                return d
        except Exception:  # noqa: BLE001 - fall back to the attributes
            pass
    out = {}
    for k, v in vars(obj).items():
        if k.startswith("__"):
            continue
        out[k] = config_dict(v) if hasattr(v, "__dict__") and not isinstance(v, type) else v
    return out


class HfTemplate:
    name = "hf_template"

    def applies_to(self, path):
        return os.path.isdir(path) and any(os.path.isfile(os.path.join(path, f))
                                           for f in ("tokenizer_config.json", "chat_template.jinja"))

    def read(self, path):
        r = ReadResult()
        tc_path = os.path.join(path, "tokenizer_config.json")
        if os.path.isfile(tc_path):
            key, ct = _first(_load_json(tc_path), ALIASES["Template"]["hf_tokenizer_config"]["chat_template"])
            where = f"{tc_path}#{key}"
            if isinstance(ct, str):
                _emit(r, "Template", lambda: Template(sha256_text(ct)), "config", where)
            elif isinstance(ct, list):
                named = {t.get("name"): t.get("template") for t in ct if isinstance(t, dict)}
                if isinstance(named.get("default"), str):
                    _emit(r, "Template", lambda: Template(sha256_text(named["default"])), "config", where + "[default]")
                others = sorted(n for n in named if n != "default")
                if others:
                    r.problems.append(f"{where}: named templates {others} besides 'default' are not in vocabulary v1")
                if "default" not in named:
                    r.problems.append(f"{where}: a list of templates without a 'default' one")
        for name in ALIASES["Template"]["hf_files"]["chat_template"]:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    text = f.read()
                _emit(r, "Template", lambda: Template(sha256_text(text)), "config", p)
        return r


# --- diffusers folders ------------------------------------------------------------------------------------------

class DiffusersConfig:
    name = "diffusers"

    def applies_to(self, path):
        return os.path.isdir(path) and any(os.path.isfile(os.path.join(path, *p)) for p in
                                           (("scheduler", "scheduler_config.json"), ("vae", "config.json")))

    def read(self, path):
        r = ReadResult()
        sk = ALIASES["Prediction"]["diffusers_scheduler"]
        sched = os.path.join(path, "scheduler", "scheduler_config.json")
        if os.path.isfile(sched):
            cfg = _load_json(sched)
            key, raw = _first(cfg, sk["kind"])
            kind = prediction_kind(raw) if key else None
            used = [key] if key else []
            if kind is None and any(m in str(cfg.get("_class_name", "")) for m in sk["flow_class_marker"]):
                kind, used = "flow", ["_class_name"]
            if kind is None and key:
                r.problems.append(f"{sched}#{key}: prediction type {raw!r} is not in vocabulary v1")
            zkey, z = _first(cfg, sk["zsnr"])
            if kind:
                zsnr = z if isinstance(z, bool) else None
                where = f"{sched}#" + ",".join(used + ([zkey] if zkey and zsnr is not None else []))
                _emit(r, "Prediction", lambda: Prediction(kind, zsnr), "config", where)
        vk = ALIASES["LatentScale"]["diffusers_vae"]
        vae = os.path.join(path, "vae", "config.json")
        if os.path.isfile(vae):
            cfg = _load_json(vae)
            key, scale = _first(cfg, vk["scale"])
            skey, shift = _first(cfg, vk["shift"])
            if key:
                where = f"{vae}#" + ",".join(x for x in (key, skey) if x)
                _emit(r, "LatentScale", lambda: LatentScale(scale, shift=shift), "config", where)
        return r


# --- single files ---------------------------------------------------------------------------------------------

def safetensors_header(path):
    """(tensor names, metadata) of a .safetensors file, reading only its header."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError("too short for a safetensors header")
        n = struct.unpack("<Q", head)[0]
        if n > min(size - 8, 100 << 20):
            raise ValueError(f"header length {n} does not fit the file: not a safetensors file")
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    return list(header), meta


class SafetensorsMeta:
    name = "safetensors"

    def applies_to(self, path):
        return str(path).endswith(".safetensors") and os.path.isfile(path)

    def read(self, path):
        r = ReadResult()
        keys, meta = safetensors_header(path)
        keys = set(keys)
        mk, kk = ALIASES["Prediction"]["safetensors_metadata"], ALIASES["Prediction"]["safetensors_keys"]
        zsnr, zsnr_from = None, None
        if any(k in keys for k in kk["zsnr"]):
            zsnr, zsnr_from = True, "key " + next(k for k in kk["zsnr"] if k in keys)
        else:
            zkey, z = _first(meta, mk["zsnr"])
            if zkey and flag(z) is not None:
                zsnr, zsnr_from = flag(z), zkey
        note = f" (zsnr from {zsnr_from})" if zsnr_from else ""
        statements = []
        key, raw = _first(meta, mk["kind"])
        if key:
            kind = prediction_kind(raw)
            if kind:
                statements.append((kind, f"__metadata__.{key}"))
            else:
                r.problems.append(f"{path}#__metadata__.{key}: prediction type {raw!r} is not in vocabulary v1")
        key, raw = _first(meta, mk["v_flag"])
        if key and flag(raw) is not None:
            statements.append(("v" if flag(raw) else "eps", f"__metadata__.{key}"))
        for k in kk["v"]:
            if k in keys:
                statements.append(("v", f"key {k}"))
        for kind, what in statements:
            _emit(r, "Prediction", lambda: Prediction(kind, zsnr), "file", f"{path}#{what}{note}")
        return r


class GgufMeta:
    name = "gguf"

    def applies_to(self, path):
        return str(path).endswith(".gguf") and os.path.isfile(path)

    def read(self, path):
        r = ReadResult()
        md = gguf.read_metadata(path)
        arch = md.get("general.architecture")
        if not isinstance(arch, str):
            r.problems.append(f"{path}: no general.architecture, so its model keys cannot be read")
            arch = "?"

        def first(names):
            return _first(md, [n.replace("{arch}", arch) for n in names])

        mk = ALIASES["ModelProps"]["gguf"]
        props, used = {}, []
        for fld, names in (("softcap", mk["softcap"]), ("sliding_window", mk["sliding_window"])):
            key, v = first(names)
            if key:
                props[fld], used = v, used + [key]
        _props(r, props, used, "file", path)
        rk = ALIASES["Rotary"]["gguf"]
        tkey, theta = first(rk["theta"])
        ykey, typ = first(rk["type"])
        if tkey or ykey:
            spec = {"rope_type": typ} if typ is not None else {}
            fkey, factor = first(rk["factor"])
            okey, omp = first(rk["original_max_position"])
            if factor is not None:
                spec["factor"] = factor
            if omp is not None:
                spec["original_max_position_embeddings"] = omp
            where = f"{path}#" + ",".join(x for x in (tkey, ykey, fkey, okey) if x)
            _rotary(r, spec, theta, tkey, where, source_kind="file")
        key, ct = first(ALIASES["Template"]["gguf"]["chat_template"])
        if isinstance(ct, str):
            _emit(r, "Template", lambda: Template(sha256_text(ct)), "file", f"{path}#{key}")
        return r


READERS = [HfConfig(), HfTemplate(), DiffusersConfig(), SafetensorsMeta(), GgufMeta()]
