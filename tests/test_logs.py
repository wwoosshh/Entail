"""Tests for where entail's lines and records go (ROADMAP M6.4, the researcher's decision of 2026-09-24): whenever entail
is on, the project's entail_logs folder - the folder the program was started from - keeps what it said; ENTAIL_LOG_DIR
moves or turns it off, ENTAIL_RECORD names the record file as before; a folder that cannot be written is said once and
the run goes on. Run: python tests/test_logs.py"""
import io
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, record, tally  # noqa: E402
from entail.contracts import RULES, Contract, Decision, Verdict  # noqa: E402

KEYS = ("ENTAIL", "ENTAIL_LOG_DIR", "ENTAIL_RECORD", "ENTAIL_VERBOSE", "ENTAIL_ON_BROKEN")


@contextmanager
def project(**env):
    """A fresh folder to start the program from, with these settings; the environment and folder restored after."""
    old_env, old_cwd = {k: os.environ.pop(k, None) for k in KEYS}, os.getcwd()
    d = tempfile.mkdtemp()
    os.chdir(d)
    os.environ.update(env)
    record._READY.clear()
    record._WARNED.clear()
    try:
        yield d
    finally:
        record.close_files()
        os.chdir(old_cwd)
        for k in KEYS:
            os.environ.pop(k, None)
            if old_env[k] is not None:
                os.environ[k] = old_env[k]


def broken():
    return Decision(Contract("load:test.lora", "test.lora_loader", ("Coverage",), ("Coverage",)), "Coverage",
                    Verdict.BROKEN, RULES["no_resolution"])


def today():
    return time.strftime("%Y-%m-%d")


def test_on_what_entail_says_is_kept_in_the_project():
    with project(ENTAIL="load") as d:
        with redirect_stdout(io.StringIO()) as out:
            load.enforce([broken()])
        folder = os.path.join(d, "entail_logs")
        log = open(os.path.join(folder, f"entail-{today()}.log"), encoding="utf-8").read()
        rec = [json.loads(x) for x in open(os.path.join(folder, f"record-{today()}.jsonl"), encoding="utf-8")]
        assert "broken at load:test.lora" in out.getvalue(), "still printed"
        assert "broken at load:test.lora" in log and f"pid {os.getpid()}" in log and today() in log
        assert rec[0]["verdict"] == "broken" and rec[0]["boundary"] == "load:test.lora"
        assert open(os.path.join(folder, ".gitignore"), encoding="utf-8").read().strip().endswith("*")
        assert os.environ["ENTAIL_LOG_DIR"] == folder, "the processes an engine starts write to the same folder"


def test_only_the_mode_set_in_code_or_off_writes_nothing():
    with project() as d:   # ENTAIL unset: tests and harnesses set the mode in code
        core.set_mode("load")
        try:
            with redirect_stdout(io.StringIO()):
                load.enforce([broken()])
        finally:
            core.set_mode("off")
        assert not os.path.exists(os.path.join(d, "entail_logs"))
    with project(ENTAIL="load", ENTAIL_LOG_DIR="off") as d:
        with redirect_stdout(io.StringIO()) as out:
            load.enforce([broken()])
        assert "broken" in out.getvalue() and not os.path.exists(os.path.join(d, "entail_logs"))


def test_the_folder_can_be_moved_and_the_record_named():
    with project(ENTAIL="load") as d:
        elsewhere, rec = os.path.join(d, "logs_here"), os.path.join(d, "mine.jsonl")
        os.environ.update(ENTAIL_LOG_DIR=elsewhere, ENTAIL_RECORD=rec)
        with redirect_stdout(io.StringIO()):
            load.enforce([broken()])
            load.say("test repair", "an engine-specific repair said this")
            tally.write_summary({"container:test": {"checks": 1}})
        log = open(os.path.join(elsewhere, f"entail-{today()}.log"), encoding="utf-8").read()
        lines = [json.loads(x) for x in open(rec, encoding="utf-8")]
        assert "broken at load:test.lora" in log and "an engine-specific repair said this" in log
        assert [("verdict" in x, "said" in x, "boundaries" in x) for x in lines] == \
            [(True, False, False), (False, True, False), (False, False, True)]
        assert not os.path.exists(os.path.join(elsewhere, f"record-{today()}.jsonl")), "ENTAIL_RECORD takes them"
        assert not os.path.exists(os.path.join(elsewhere, ".gitignore")), "a folder the user named is left as it is"


def test_a_process_started_later_writes_to_the_same_folder():
    """An engine runs its model in processes it starts itself, from wherever: they write where the program started."""
    import subprocess

    with project(ENTAIL="load") as d:
        folder = record.log_dir()   # the start-up hook fixes it when the program starts
        elsewhere = tempfile.mkdtemp()
        code = ("import sys; sys.path.insert(0, sys.argv[1]); from entail import load; "
                "load.say('child', 'said in a process started later, elsewhere')")
        subprocess.run([sys.executable, "-c", code, os.path.dirname(HERE)], cwd=elsewhere, env=dict(os.environ),
                       check=True, capture_output=True)
        log = open(os.path.join(folder, f"entail-{today()}.log"), encoding="utf-8").read()
        assert "said in a process started later, elsewhere" in log
        assert not os.path.exists(os.path.join(elsewhere, "entail_logs")) and folder.startswith(d)


def test_the_files_are_kept_open_and_every_line_lands_at_once():
    """One open per file per process, a flush per line (M17.3's S4 run: an open and close per line cost 4.6 ms on a
    9P mount, 6% of a batch-32 decode with a per-request boundary); closed, a file is reopened for appending."""
    with project(ENTAIL="load") as d:
        with redirect_stdout(io.StringIO()):
            load.enforce([broken()])
            load.enforce([broken()])
        folder = os.path.join(d, "entail_logs")
        rec_path = os.path.join(folder, f"record-{today()}.jsonl")
        assert sorted(k[1] for k in record._OPEN) == sorted([os.path.join(folder, f"entail-{today()}.log"), rec_path])
        assert all(k[0] == os.getpid() for k in record._OPEN) and all(not f.closed for f in record._OPEN.values())
        assert len(open(rec_path, encoding="utf-8").read().splitlines()) == 2, "each line is flushed as it is written"
        record.close_files()
        assert record._OPEN == {}
        with redirect_stdout(io.StringIO()):
            load.enforce([broken()])
        assert len(open(rec_path, encoding="utf-8").read().splitlines()) == 3 and len(record._OPEN) == 2


def test_a_folder_that_cannot_be_written_is_said_once_and_the_run_goes_on():
    with project(ENTAIL="load") as d:
        blocked = os.path.join(d, "a_file")
        open(blocked, "w").close()
        os.environ["ENTAIL_LOG_DIR"] = os.path.join(blocked, "entail_logs")   # under a file: cannot be made
        err = io.StringIO()
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(err):
            load.enforce([broken()])
            load.enforce([broken()])
        assert out.getvalue().count("broken at load:test.lora") == 2, "the lines still reach the console"
        assert err.getvalue().count("could not write") == 2, "once for the log file and once for the record file"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
