# python-ml-primitives-correctness-lab

A very small deterministic Python correctness lab about manual matrix-vector multiplication, shape contracts, numpy.matmul comparison, numerically stable softmax, naive softmax overflow, softmax shift invariance, exact vs approximate floating-point equality, and the limits of tiny implementation exercises.

This is an educational correctness lab. It is not an interview benchmark, a machine-learning benchmark, a numerical-library replacement, a statistical-quality test, a model evaluation, a performance benchmark, or a claim that implementing primitives is sufficient to understand their mathematical theory.

No model training. No dataset downloads. No GPUs. Only Python standard library + NumPy.

## Hacker News thread access

The relevant public thread evidence was captured before the discussion summary was written.

```
hackernews get-item --id 40925896
```

See `hn_thread_evidence.md` and `hn_comments_sanitized.json`.

## Discussion (from HN thread https://news.ycombinator.com/item?id=40925896)

- **TrackerFF** argued that professionals generally should not replace established numerical libraries with inferior hand-written implementations, and questioned using elementary numerical routines as interview filters. They noted such solutions are often subpar compared to industry standard packages, and that failing to recall formulas from memory does not establish lack of competence (e.g., SVM equations after 10 years).

- **mchab** (the site author) said the project was intended as a learning tool inspired by learning through building (Andrej Karpathy's zero-to-hero videos), rather than as interview preparation. They acknowledged the original "Leetcode for ML" title triggered a lot of people.

- **jvanderbot** said implementing matrix-factorization routines helped them understand library design and performance decisions, including SIMD speedups.

- **coliveira** argued that writing an implementation does not by itself establish understanding of the underlying mathematics.

- **bArray** viewed the exercises as a useful way to check understanding of fundamentals.

- **esafak** said the difficult part of machine learning is often theory rather than implementation.

- **oehpr** replied that an implementation exercise can provide motivation and a test of understanding.

- **drfunk** raised correctness problems around numerical comparisons and suggested `numpy.testing.assert_allclose`.

- **diimdeep** preferred ordinary local markdown and python files over a siloed exercise website.

- **ZoomerCretin** said an example's matrix-vector notation was unclear.

- **r-zip** treated matrix multiplication as a basic machine-learning equivalent of fizzbuzz.

The thread does not prove that implementing primitives is the best way for every person to learn mathematics, that inability to reproduce a formula from memory means someone is unqualified for machine-learning work, that a tiny python exercise predicts job performance, that hand-written routines should replace numpy in production, that passing a few numerical examples proves an implementation is generally correct, that approximately equal arrays are interchangeable for every application, that finite softmax output proves model correctness, or that this lab evaluates interviewing practices.

## What this lab does

- `manual_matvec(matrix, vector)`: explicit Python loops, strict shape contracts, float64.
- `stable_softmax(values)`: max-shift before exp, finite-output checked.
- Compares manual_matvec with `numpy.matmul` on a fixed 2×2 example.
- Verifies shape-mismatch rejection.
- Verifies softmax shift invariance: softmax([1000,1001,1002]) ≈ softmax([0,1,2]).
- Shows naive exp overflow on large logits.
- Shows exact equality vs `numpy.testing.assert_allclose` for `0.1+0.2` vs `0.3`.

## Results

See `RESULTS.md`.

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
