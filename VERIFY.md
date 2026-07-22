# VERIFY

Repository: https://github.com/necat101/python-ml-primitives-correctness-lab

Implementation SHA: 941bb3699ad30a60ce58a63ddf2801eb269af2e8

## Clone / checkout

```
git clone https://github.com/necat101/python-ml-primitives-correctness-lab.git /clean-checkout/repo
cd /clean-checkout/repo
git checkout 941bb3699ad30a60ce58a63ddf2801eb269af2e8
```

## Environment

- Python: 3.12.3 (CPython)
- OS family: Linux
- NumPy available: True
- NumPy version: 2.4.6

## Commands executed

1. `python3 -m py_compile run_lab.py test_lab.py`
   - exit code: 0

2. `python3 run_lab.py`
   - exit code: 0

3. `python3 -m unittest -v`
   - exit code: 0

## Unittest results

- Discovered: 24
- Executed: 24
- Skipped: 0
- Failed: 0

## Lab counts

- Cases: 7
- Methods: 3
- Rows: 21

## Classification buckets

- context_only: 4
- dependency_skip: 0
- expected_close: 2
- expected_equal: 1
- expected_nonfinite: 1
- expected_rejection: 1
- fail: 0
- local_observation: 7
- not_applicable: 0
- pass: 5

Dependency skips: 0
Failures: 0

## Regeneration checks

- observations.json comparison: bit-identical
- RESULTS.md comparison: bit-identical
- artifact scanner result: OK
- git diff --exit-code: 0
- git status --porcelain: (empty)
