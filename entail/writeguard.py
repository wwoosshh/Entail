"""writeguard: between operations, nothing may write to what a producer issued until its consumer has read it (ROADMAP
M19 L5.4e, approach A: between the operations; ENTAIL=structure_writes).

The structural check (kernel_ir) proves what the consumer kernel computes from its operands. Whether the operands
still hold what the producers made is a question about the writes that happen between the producer and the consumer.
Here those writes are refused before they run, and no kernel is changed:

  ranges     every tensor a producer issues (the activation and its scale until the gate consumed them, the weight
             and its scale for as long as they live) is kept as a byte range of device memory (`register`).
  ATen ops   a Python dispatch mode (`Guard`, pushed on the thread that issues) sees every PyTorch operation; one
             whose schema says it writes an argument that overlaps a live range is refused before it runs (`write`).
             The other arguments, and operations that write nothing, pass.
  Triton     the launch hook (adapters/vllm_block_fp8_guarantee.install_triton) offers every Triton launch: one that
             writes - by its own TTIR, kernel_ir.written_args - through an argument overlapping a live range, and
             that is not the gate's own consumer launch, is refused (`check_launch`); reading passes. A kernel whose
             writes cannot be followed back to its arguments counts as writing through all of them.
  graphs     a CUDA graph is checked while it is captured: Python runs then, the same refusals apply, and a refused
             write fails the capture. What replays is what was captured.

What it cannot see, by construction: a write through an address that is not in the operation's arguments (a kernel
writing past the end of its own buffer), a C++ or CUDA operation whose schema does not say that it writes, and
anything that runs outside PyTorch's dispatcher and Triton's launcher. Those are not refused here (L5.4e cases I4,
I8 measure them).
"""
import bisect
import threading
import weakref

_LOCK = threading.Lock()
_STARTS = []               # sorted start addresses of the live ranges
_RANGES = {}               # start -> (end, serial, role)
_BY_SERIAL = {}            # serial -> start
_WRITES = {}               # OpOverload -> [(argument position, name)] its schema says it writes
_LOCAL = threading.local()
_STATS = {}


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


def span(t):
    """(start, end) of the device bytes a tensor addresses, or None (a host tensor, no elements)."""
    try:
        if not t.is_cuda or t.numel() == 0:
            return None
        extent = 0
        for size, stride in zip(t.shape, t.stride()):
            if size > 1:
                extent += (size - 1) * abs(stride)
        start = t.data_ptr()
        return start, start + (extent + 1) * t.element_size()
    except Exception:  # noqa: BLE001 - a tensor without storage (meta, fake): nothing to compare
        return None


def overlapping(start, end):
    """The (serial, role) of a live range that [start, end) overlaps, or None. The live ranges never overlap each
    other, so only the last one that starts before `end` can."""
    i = bisect.bisect_left(_STARTS, end) - 1
    if i < 0:
        return None
    e, serial, role = _RANGES[_STARTS[i]]
    return (serial, role) if e > start else None


def register(t, serial, role):
    """Keep the bytes of an issued tensor as a live range until `release(serial)` or until the tensor is collected."""
    sp = span(t)
    if sp is None:
        return
    with _LOCK:
        # a range the new one overlaps belongs to memory issued before and no longer what it was (a buffer used
        # again): the new issue replaces it
        while True:
            hit = overlapping(*sp)
            if hit is None:
                break
            s = _BY_SERIAL.pop(hit[0], None)
            if s is None:
                s = next(k for k, v in _RANGES.items() if v[1] == hit[0])
            del _RANGES[s]
            _STARTS.pop(bisect.bisect_left(_STARTS, s))
        bisect.insort(_STARTS, sp[0])
        _RANGES[sp[0]] = (sp[1], serial, role)
        _BY_SERIAL[serial] = sp[0]
    weakref.finalize(t, release, serial)
    ensure_mode()
    _count("registered")


def release(serial):
    with _LOCK:
        s = _BY_SERIAL.pop(serial, None)
        if s is None:
            return
        cur = _RANGES.get(s)
        if cur is not None and cur[1] == serial:
            del _RANGES[s]
            i = bisect.bisect_left(_STARTS, s)
            if i < len(_STARTS) and _STARTS[i] == s:
                _STARTS.pop(i)
    _count("released")


def live() -> int:
    return len(_RANGES)


def _refuse(what, hit):
    from . import guarantee

    serial, role = hit
    _count("refused")
    guarantee.refuse_write(f"{what} writes to the bytes of the {role} issued as {serial}, between its producer and "
                           f"its consumer", {"writer": what, "issue": serial, "role": role})


def _written(func):
    w = _WRITES.get(func)
    if w is None:
        w = []
        try:
            for i, a in enumerate(func._schema.arguments):
                if a.alias_info is not None and a.alias_info.is_write:
                    w.append((i, a.name))
        except Exception:  # noqa: BLE001 - no schema to read: nothing is said to be written
            pass
        _WRITES[func] = w
    return w


def check_args(what, tensors):
    """Refuse when one of `tensors` (written ones) overlaps a live range. Anything that is not a device tensor has
    no span and passes."""
    for t in tensors:
        if isinstance(t, (list, tuple)):
            check_args(what, t)
            continue
        sp = span(t)
        if sp is not None:
            hit = overlapping(*sp)
            if hit is not None:
                _refuse(what, hit)


def check_launch(name, tensors, allowed=()):
    """A Triton launch: refuse when a tensor argument other than the ones in `allowed` (the gate's own operands)
    overlaps a live range. A launch cannot say which of its pointers it only reads."""
    if not _RANGES:
        return
    keep = {id(x) for x in allowed}
    for t in tensors:
        if id(t) not in keep:
            sp = span(t)
            if sp is not None:
                hit = overlapping(*sp)
                if hit is not None:
                    _refuse(f"the Triton kernel {name}", hit)
    _count("launches_checked")


def _guard_class():
    from torch.utils._python_dispatch import TorchDispatchMode

    class Guard(TorchDispatchMode):
        """Refuses a PyTorch operation that writes into a live range, before it runs."""

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if _RANGES:
                w = _written(func)
                if w:
                    check_args(str(func), [args[i] if i < len(args) else kwargs.get(n) for i, n in w])
            return func(*args, **kwargs)

    return Guard


_GUARD = None


def ensure_mode():
    """Push the dispatch mode on this thread, once."""
    global _GUARD
    if getattr(_LOCAL, "pushed", False):
        return
    try:
        from torch.utils._python_dispatch import _push_mode

        if _GUARD is None:
            _GUARD = _guard_class()
        _push_mode(_GUARD())
        _LOCAL.pushed = True
        _count("modes_pushed")
    except Exception:  # noqa: BLE001 - no mode: ATen writes are not seen on this thread (recorded)
        _count("mode_push_failed")


def stats() -> dict:
    out = {f"writes_{k}": v for k, v in _STATS.items()}
    out["writes_live_ranges"] = len(_RANGES)
    return out


def reset() -> None:
    with _LOCK:
        _STARTS.clear()
        _RANGES.clear()
        _BY_SERIAL.clear()
    _STATS.clear()
