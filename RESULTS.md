# RESULTS

Python: 3.12.3 (CPython)
OS: Linux
NumPy available: True
NumPy version: 2.4.6

Cases: 7
Methods: 3
Rows: 21

## Classification counts

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

## Observations

### Matrix-vector reference
manual_matvec result: [17.0, 39.0]
numpy.matmul result: [17.0, 39.0]
Classification: expected_equal

### Shape contract
manual_rejected=True, numpy_rejected=True
Classification: expected_rejection

### Stable softmax shift invariance
output_logits: [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
output_shifted: [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
finite=True, close_match=True, input_unmutated=True
Classification: expected_close

### Naive softmax overflow
naive_has_nonfinite=True, stable_all_finite=True, stable_sum_close_to_one=True
Classification: expected_nonfinite

### Floating comparison policy
exact_equal=False, close_ok=True
Classification: expected_close

## Evidence separation

Hacker News commenters argued:
- TrackerFF: professionals generally should not replace established numerical libraries with inferior hand-written implementations; questioned using elementary numerical routines as interview filters.
- mchab: the project was intended as a learning tool inspired by learning through building (Karpathy zero-to-hero), rather than as interview preparation.
- jvanderbot: implementing matrix-factorization routines helped them understand library design and performance decisions.
- coliveira: writing an implementation does not by itself establish understanding of the underlying mathematics.
- bArray: the exercises are a useful way to check understanding of fundamentals.
- esafak: the difficult part of machine learning is often theory rather than implementation.
- oehpr: an implementation exercise can provide motivation and a test of understanding.
- drfunk: correctness problems around numerical comparisons; suggested numpy.testing.assert_allclose.
- diimdeep: preferred ordinary local markdown and python files over a siloed exercise website.
- ZoomerCretin: an example's matrix-vector notation was unclear.
- r-zip: treated matrix multiplication as a basic machine-learning equivalent of fizzbuzz.

Linked exercise site (https://www.deep-ml.com/): presents small ML coding exercises in a browser.

Official NumPy documentation specifies:
- numpy.matmul: matrix multiplication with shape-contract checking.
- numpy.testing.assert_allclose: approximate floating-point comparison with rtol/atol.
- numpy.errstate: control floating-point error handling.
- numpy.exp: exponential function.
- numpy.isfinite: finite-value checks.

Local environment reported:
- Python 3.12.3 (CPython)
- NumPy available=True, version=2.4.6

Lab directly observed:
- manual_matvec agrees with numpy.matmul for the fixed 2×2 example.
- Both implementations reject the shape mismatch.
- stable_softmax shift invariance holds approximately for the fixed inputs.
- Naive softmax overflows on [1000,1001,1002]; stable version is finite and sums ≈1.
- Exact equality differs from approximate equality for [0.1+0.2, 1.0] vs [0.3, 1.0].

Lab did not test:
- Other dtypes, batch shapes, sparse inputs, complex values, NaN/Inf handling, accelerators, or production workloads.
- Statistical quality, model evaluation, training, or inference performance.
- Interview prediction, learning effectiveness, or theoretical understanding.

## Non-claims

This repository does not prove:
- that implementing numerical primitives is the best learning method for everyone;
- that implementation ability establishes theoretical understanding;
- that inability to implement a primitive from memory establishes lack of competence;
- that this lab predicts interview or job performance;
- that manual_matvec should replace numpy.matmul;
- that stable_softmax is a complete production implementation;
- that passing twenty-one rows proves general correctness;
- that approximate equality is appropriate for every application;
- that finite softmax values validate a model;
- that the linked exercise site is correct or incorrect as a whole;
- or that the lab is production-ready.

