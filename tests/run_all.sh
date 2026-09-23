#!/bin/bash
# All entail unit tests:  bash tests/run_all.sh
# PYTHON picks the interpreter (default: python). Tests that need a local model look in ENTAIL_TEST_MODELS
# (default ~/models) and skip when the model is not there.
# A file passes only if python exits 0: a crash after some "ok" lines is a failure, not a partial pass.
set -u
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
fail=0
for t in tests/test_*.py; do
  out=$("$PY" "$t" 2>&1)
  rc=$?
  n=$(printf '%s\n' "$out" | grep -c '^ok ')
  if [ $rc -eq 0 ] && [ "$n" -gt 0 ]; then
    echo "PASS $t ($n checks)"
  else
    echo "FAIL $t (exit $rc, $n ok before it)"; printf '%s\n' "$out" | tail -8; fail=1
  fi
done
exit $fail
