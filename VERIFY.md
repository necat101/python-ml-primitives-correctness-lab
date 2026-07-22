# VERIFY

Repository: https://github.com/necat101/python-ml-primitives-correctness-lab

Implementation SHA: 618be575483d27d5ec05ebfb0d89552c2b0dfc1c

## Clone / checkout

```
git clone https://github.com/necat101/python-ml-primitives-correctness-lab.git /clean-checkout/repo
cd /clean-checkout/repo
git checkout 618be575483d27d5ec05ebfb0d89552c2b0dfc1c
```

## Environment

- Python: 3.12.3 (CPython)
- OS family: Linux
- NumPy available: True
- NumPy version: 2.4.6

## Commands executed (clean checkout, implementation SHA)

1. `python -m py_compile run_lab.py test_lab.py`
   - exit code: 0

2. `python run_lab.py`
   - exit code: 0

3. `python -m unittest -v`
   - exit code: 0

## Unittest results (clean checkout)

- Discovered: 26
- Executed: 26
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

## Regeneration checks (clean checkout)

- observations.json comparison: bit-identical
- RESULTS.md comparison: bit-identical
- artifact scanner result: OK
- git diff --exit-code: 0
- git status --porcelain: (empty)

## Narrow conclusions

- manual_matvec([[1,2],[3,4]], [5,6]) == [17.0, 39.0] == numpy.matmul result for this fixed input.
- Both manual_matvec and numpy.matmul reject the shape-mismatch input (2×3 matrix, length-2 vector).
- stable_softmax([1000,1001,1002]) is approximately equal to stable_softmax([0,1,2]), finite, sums ≈1.0, input not mutated.
- Naive exp([1000,1001,1002]) / sum(exp(...)) produces nonfinite values; stable_softmax produces finite values summing ≈1.0.
- numpy.array_equal([0.1+0.2, 1.0], [0.3, 1.0]) is False; numpy.testing.assert_allclose(..., rtol=1e-12, atol=1e-12) is True.

These observations are scoped to the fixed inputs in this repository. They do not establish general correctness, production readiness, interview predictive validity, or theoretical understanding.

## Post-VERIFY scanner run

After writing VERIFY.md, the unittest suite was rerun with VERIFY.md present:

- Command: `python -m unittest -v`
- Discovered: 26
- Executed: 26
- Skipped: 0
- Failed: 0
- artifact_scanner: OK (includes VERIFY.md)
