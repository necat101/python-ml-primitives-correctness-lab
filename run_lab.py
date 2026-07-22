#!/usr/bin/env python3
import json
import sys
import platform

CASE_IDS = [
    "runtime_and_dependency_marker",
    "matrix_vector_reference_marker",
    "matrix_vector_shape_contract_marker",
    "stable_softmax_shift_invariance_marker",
    "naive_softmax_overflow_marker",
    "floating_comparison_policy_marker",
    "no_learning_interview_or_production_validity_claim_marker",
]
METHOD_IDS = ["inspect_inputs", "execute_case", "verify_relation"]

# load inputs / tolerances from cases.json if present
try:
    with open("cases.json", "r", encoding="utf-8") as f:
        cases_doc = json.load(f)
    INPUTS = cases_doc.get("inputs", {})
    TOL = cases_doc.get("tolerances", {})
    RTOL = float(TOL.get("rtol", 1e-12))
    ATOL = float(TOL.get("atol", 1e-12))
except Exception:
    INPUTS = {}
    RTOL = 1e-12
    ATOL = 1e-12

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NUMPY_VERSION = np.__version__
except Exception:
    NUMPY_AVAILABLE = False
    NUMPY_VERSION = None
    np = None

# --- production functions ---

def manual_matvec(matrix, vector):
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy not available")
    m = np.asarray(matrix, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if v.ndim != 1:
        raise ValueError("vector must be 1-D")
    if m.shape[1] != v.shape[0]:
        raise ValueError("shape contract violated: matrix columns != vector length")
    out = np.zeros(m.shape[0], dtype=np.float64)
    for i in range(m.shape[0]):
        acc = 0.0
        for j in range(m.shape[1]):
            acc += float(m[i, j]) * float(v[j])
        out[i] = acc
    return out


def stable_softmax(values):
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy not available")
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("input must be 1-D")
    if x.size == 0:
        raise ValueError("input must be nonempty")
    # do not mutate caller input: x is a copy due to asarray with dtype
    shift = np.max(x)
    e = np.exp(x - shift)
    s = np.sum(e)
    return e / s


# --- case handlers ---

def handle_runtime_and_dependency_marker(method):
    obs = {}
    if method == "inspect_inputs":
        return "local_observation", obs
    if method == "execute_case":
        obs = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "os_family": platform.system(),
            "numpy_available": NUMPY_AVAILABLE,
            "numpy_version": NUMPY_VERSION,
            "case_count": len(CASE_IDS),
            "method_count": len(METHOD_IDS),
            "row_count": len(CASE_IDS) * len(METHOD_IDS),
            "rtol": RTOL,
            "atol": ATOL,
            "fixed_input_names": ["matrix_vector_reference", "matrix_vector_shape_mismatch", "softmax_logits", "softmax_shifted", "floating_comparison"],
        }
        return "local_observation", obs
    if method == "verify_relation":
        return "context_only", obs
    return "fail", {"reason": "unknown method"}

def handle_matrix_vector_reference_marker(method):
    if not NUMPY_AVAILABLE:
        return "dependency_skip", {"reason": "numpy not available"}
    ref = INPUTS.get("matrix_vector_reference", {})
    matrix = ref.get("matrix", [[1.0, 2.0], [3.0, 4.0]])
    vector = ref.get("vector", [5.0, 6.0])
    expected = ref.get("expected", [17.0, 39.0])
    if method == "inspect_inputs":
        return "local_observation", {"matrix": matrix, "vector": vector, "expected": expected}
    if method == "execute_case":
        try:
            manual = manual_matvec(matrix, vector)
            numpy_result = np.matmul(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
            obs = {
                "matrix": matrix,
                "vector": vector,
                "manual_result": manual.tolist(),
                "numpy_result": numpy_result.tolist(),
                "expected_result": expected,
                "manual_shape": list(manual.shape),
                "numpy_shape": list(numpy_result.shape),
                "manual_dtype": str(manual.dtype),
                "numpy_dtype": str(numpy_result.dtype),
            }
            return "pass", obs
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    if method == "verify_relation":
        try:
            manual = manual_matvec(matrix, vector)
            numpy_result = np.matmul(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
            expected_arr = np.asarray(expected, dtype=np.float64)
            shape_match = manual.shape == numpy_result.shape
            exact_equal_manual_expected = np.array_equal(manual, expected_arr)
            exact_equal_manual_numpy = np.array_equal(manual, numpy_result)
            obs = {
                "shape_match": bool(shape_match),
                "exact_equal_manual_expected": bool(exact_equal_manual_expected),
                "exact_equal_manual_numpy": bool(exact_equal_manual_numpy),
                "manual_result": manual.tolist(),
                "numpy_result": numpy_result.tolist(),
            }
            if shape_match and exact_equal_manual_expected and exact_equal_manual_numpy:
                return "expected_equal", obs
            return "fail", {"reason": "relation not met", **obs}
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    return "fail", {"reason": "unknown method"}

def handle_matrix_vector_shape_contract_marker(method):
    if not NUMPY_AVAILABLE:
        return "dependency_skip", {"reason": "numpy not available"}
    ref = INPUTS.get("matrix_vector_shape_mismatch", {})
    matrix = ref.get("matrix", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vector = ref.get("vector", [7.0, 8.0])
    if method == "inspect_inputs":
        return "local_observation", {"matrix": matrix, "vector": vector}
    if method == "execute_case":
        manual_rejected = False
        manual_exc = None
        try:
            manual_matvec(matrix, vector)
        except Exception as e:
            manual_rejected = True
            manual_exc = type(e).__name__
        numpy_rejected = False
        numpy_exc = None
        try:
            np.matmul(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
        except Exception as e:
            numpy_rejected = True
            numpy_exc = type(e).__name__
        obs = {
            "manual_rejected": manual_rejected,
            "manual_exception": manual_exc,
            "numpy_rejected": numpy_rejected,
            "numpy_exception": numpy_exc,
            "reason": "shape contract"
        }
        return "pass", obs
    if method == "verify_relation":
        manual_rejected = False
        try:
            manual_matvec(matrix, vector)
        except ValueError:
            manual_rejected = True
        except Exception:
            manual_rejected = False
        numpy_rejected = False
        try:
            np.matmul(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
        except Exception:
            numpy_rejected = True
        obs = {"manual_rejected": manual_rejected, "numpy_rejected": numpy_rejected}
        if manual_rejected and numpy_rejected:
            return "expected_rejection", obs
        return "fail", {"reason": "rejection mismatch", **obs}
    return "fail", {"reason": "unknown method"}

def handle_stable_softmax_shift_invariance_marker(method):
    if not NUMPY_AVAILABLE:
        return "dependency_skip", {"reason": "numpy not available"}
    logits = INPUTS.get("softmax_logits", [1000.0, 1001.0, 1002.0])
    shifted = INPUTS.get("softmax_shifted", [0.0, 1.0, 2.0])
    if method == "inspect_inputs":
        return "local_observation", {"logits": logits, "shifted": shifted, "rtol": RTOL, "atol": ATOL}
    if method == "execute_case":
        try:
            logits_arr = np.asarray(logits, dtype=np.float64)
            shifted_arr = np.asarray(shifted, dtype=np.float64)
            logits_copy = logits_arr.copy()
            shifted_copy = shifted_arr.copy()
            out1 = stable_softmax(logits_arr)
            out2 = stable_softmax(shifted_arr)
            obs = {
                "logits": logits,
                "shifted": shifted,
                "output_logits": out1.tolist(),
                "output_shifted": out2.tolist(),
                "output1_finite": bool(np.all(np.isfinite(out1))),
                "output2_finite": bool(np.all(np.isfinite(out2))),
                "sum1": float(np.sum(out1)),
                "sum2": float(np.sum(out2)),
                "input_logits_mutated": bool(not np.array_equal(logits_arr, logits_copy)),
                "input_shifted_mutated": bool(not np.array_equal(shifted_arr, shifted_copy)),
            }
            return "pass", obs
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    if method == "verify_relation":
        try:
            logits_arr = np.asarray(logits, dtype=np.float64)
            shifted_arr = np.asarray(shifted, dtype=np.float64)
            logits_copy = logits_arr.copy()
            shifted_copy = shifted_arr.copy()
            out1 = stable_softmax(logits_arr)
            out2 = stable_softmax(shifted_arr)
            finite1 = np.all(np.isfinite(out1))
            finite2 = np.all(np.isfinite(out2))
            sum1_close = abs(float(np.sum(out1)) - 1.0) <= ATOL + RTOL * 1.0
            sum2_close = abs(float(np.sum(out2)) - 1.0) <= ATOL + RTOL * 1.0
            try:
                np.testing.assert_allclose(out1, out2, rtol=RTOL, atol=ATOL)
                close_match = True
            except AssertionError:
                close_match = False
            input_unmutated = np.array_equal(logits_arr, logits_copy) and np.array_equal(shifted_arr, shifted_copy)
            obs = {
                "finite": bool(finite1 and finite2),
                "sum1_close_to_one": bool(sum1_close),
                "sum2_close_to_one": bool(sum2_close),
                "close_match": close_match,
                "input_unmutated": bool(input_unmutated),
                "output_logits": out1.tolist(),
                "output_shifted": out2.tolist(),
            }
            if finite1 and finite2 and sum1_close and sum2_close and close_match and input_unmutated:
                return "expected_close", obs
            return "fail", {"reason": "relation not met", **obs}
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    return "fail", {"reason": "unknown method"}

def handle_naive_softmax_overflow_marker(method):
    if not NUMPY_AVAILABLE:
        return "dependency_skip", {"reason": "numpy not available"}
    logits = INPUTS.get("softmax_logits", [1000.0, 1001.0, 1002.0])
    if method == "inspect_inputs":
        return "local_observation", {"logits": logits}
    if method == "execute_case":
        try:
            arr = np.asarray(logits, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                exp_naive = np.exp(arr)
                sum_naive = np.sum(exp_naive)
                naive = exp_naive / sum_naive
            stable = stable_softmax(arr)
            obs = {
                "naive_result": [None if not np.isfinite(x) else float(x) for x in naive],
                "naive_finite_mask": [bool(np.isfinite(x)) for x in naive],
                "stable_result": stable.tolist(),
                "stable_finite_mask": [True] * len(stable),
                "stable_sum": float(np.sum(stable)),
            }
            return "pass", obs
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    if method == "verify_relation":
        try:
            arr = np.asarray(logits, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                exp_naive = np.exp(arr)
                sum_naive = np.sum(exp_naive)
                naive = exp_naive / sum_naive
            stable = stable_softmax(arr)
            naive_has_nonfinite = not bool(np.all(np.isfinite(naive)))
            stable_all_finite = bool(np.all(np.isfinite(stable)))
            stable_sum_close = abs(float(np.sum(stable)) - 1.0) <= ATOL + RTOL * 1.0
            obs = {
                "naive_has_nonfinite": naive_has_nonfinite,
                "stable_all_finite": stable_all_finite,
                "stable_sum_close_to_one": stable_sum_close,
                "naive_finite_mask": [bool(np.isfinite(x)) for x in naive],
            }
            if naive_has_nonfinite and stable_all_finite and stable_sum_close:
                return "expected_nonfinite", obs
            return "fail", {"reason": "relation not met", **obs}
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    return "fail", {"reason": "unknown method"}

def handle_floating_comparison_policy_marker(method):
    if not NUMPY_AVAILABLE:
        return "dependency_skip", {"reason": "numpy not available"}
    fc = INPUTS.get("floating_comparison", {})
    left = fc.get("left", [0.1 + 0.2, 1.0])
    right = fc.get("right", [0.3, 1.0])
    if method == "inspect_inputs":
        return "local_observation", {"left": left, "right": right, "left_expr": "0.1 + 0.2", "right_expr": "0.3"}
    if method == "execute_case":
        try:
            l = np.asarray(left, dtype=np.float64)
            r = np.asarray(right, dtype=np.float64)
            exact = np.array_equal(l, r)
            try:
                np.testing.assert_allclose(l, r, rtol=RTOL, atol=ATOL)
                close_ok = True
            except AssertionError:
                close_ok = False
            obs = {"left": left, "right": right, "exact_equal": bool(exact), "close_ok": close_ok}
            return "pass", obs
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    if method == "verify_relation":
        try:
            l = np.asarray(left, dtype=np.float64)
            r = np.asarray(right, dtype=np.float64)
            exact = np.array_equal(l, r)
            try:
                np.testing.assert_allclose(l, r, rtol=RTOL, atol=ATOL)
                close_ok = True
            except AssertionError:
                close_ok = False
            obs = {"exact_equal": bool(exact), "close_ok": close_ok}
            if (not exact) and close_ok:
                return "expected_close", obs
            return "fail", {"reason": "relation not met", **obs}
        except Exception as e:
            return "fail", {"reason": type(e).__name__}
    return "fail", {"reason": "unknown method"}

def handle_no_learning_interview_or_production_validity_claim_marker(method):
    # context only – always
    return "context_only", {}

HANDLERS = {
    "runtime_and_dependency_marker": handle_runtime_and_dependency_marker,
    "matrix_vector_reference_marker": handle_matrix_vector_reference_marker,
    "matrix_vector_shape_contract_marker": handle_matrix_vector_shape_contract_marker,
    "stable_softmax_shift_invariance_marker": handle_stable_softmax_shift_invariance_marker,
    "naive_softmax_overflow_marker": handle_naive_softmax_overflow_marker,
    "floating_comparison_policy_marker": handle_floating_comparison_policy_marker,
    "no_learning_interview_or_production_validity_claim_marker": handle_no_learning_interview_or_production_validity_claim_marker,
}

VALID_CLASSIFICATIONS = {"pass","expected_equal","expected_close","expected_rejection","expected_nonfinite","local_observation","context_only","dependency_skip","not_applicable","fail"}

def _load_expected_map(expected_map_override=None):
    if expected_map_override is not None:
        return expected_map_override
    try:
        with open("cases.json","r",encoding="utf-8") as f:
            cases_doc = json.load(f)
        return cases_doc.get("expected_classifications", {})
    except Exception:
        return {}

def build_rows(expected_map_override=None):
    rows = []
    expected_map = _load_expected_map(expected_map_override)
    for case_id in CASE_IDS:
        handler = HANDLERS.get(case_id)
        for method_id in METHOD_IDS:
            key = f"{case_id}:{method_id}"
            expected_classification = expected_map.get(key)
            if handler is None:
                actual_classification = "fail"
                observation = {"reason": "missing handler"}
            else:
                try:
                    result = handler(method_id)
                    # handler must return a 2-tuple
                    if not isinstance(result, tuple) or len(result) != 2:
                        actual_classification = "fail"
                        observation = {"reason": "missing classification"}
                    else:
                        actual_classification, observation = result
                        if actual_classification not in VALID_CLASSIFICATIONS:
                            actual_classification = "fail"
                            observation = {"reason": "invalid classification"}
                except Exception as e:
                    actual_classification = "fail"
                    observation = {"reason": type(e).__name__}
            rows.append({
                "case_id": case_id,
                "method_id": method_id,
                "expected_classification": expected_classification,
                "actual_classification": actual_classification,
                "observation": observation,
            })
    return rows

def main():
    rows = build_rows()
    # write observations.json
    with open("observations.json","w",encoding="utf-8",newline="\n") as f:
        json.dump(rows, f, indent=2)
    # counts
    from collections import Counter
    counts = Counter(r["actual_classification"] for r in rows)
    # helper to get row
    def get_row(case, method):
        for r in rows:
            if r["case_id"]==case and r["method_id"]==method:
                return r
        return None
    def get_obs(case, method):
        row = get_row(case, method)
        return row["observation"] if row else {}
    def get_actual(case, method):
        row = get_row(case, method)
        return row["actual_classification"] if row else "fail"

    mv_row = get_row("matrix_vector_reference_marker","verify_relation")
    mv_obs = mv_row["observation"] if mv_row else {}
    mv_actual = mv_row["actual_classification"] if mv_row else "fail"

    shape_row = get_row("matrix_vector_shape_contract_marker","verify_relation")
    shape_obs = shape_row["observation"] if shape_row else {}
    shape_actual = shape_row["actual_classification"] if shape_row else "fail"

    softmax_row = get_row("stable_softmax_shift_invariance_marker","verify_relation")
    softmax_obs = softmax_row["observation"] if softmax_row else {}
    softmax_actual = softmax_row["actual_classification"] if softmax_row else "fail"

    naive_row = get_row("naive_softmax_overflow_marker","verify_relation")
    naive_obs = naive_row["observation"] if naive_row else {}
    naive_actual = naive_row["actual_classification"] if naive_row else "fail"

    float_row = get_row("floating_comparison_policy_marker","verify_relation")
    float_obs = float_row["observation"] if float_row else {}
    float_actual = float_row["actual_classification"] if float_row else "fail"

    lines = []
    lines.append("# RESULTS")
    lines.append("")
    lines.append(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    lines.append(f"OS: {platform.system()}")
    lines.append(f"NumPy available: {NUMPY_AVAILABLE}")
    if NUMPY_AVAILABLE:
        lines.append(f"NumPy version: {NUMPY_VERSION}")
    lines.append("")
    lines.append(f"Cases: {len(CASE_IDS)}")
    lines.append(f"Methods: {len(METHOD_IDS)}")
    lines.append(f"Rows: {len(rows)}")
    lines.append("")
    lines.append("## Classification counts")
    lines.append("")
    for cls in sorted(VALID_CLASSIFICATIONS):
        lines.append(f"- {cls}: {counts.get(cls,0)}")
    lines.append("")
    lines.append("## Observations")
    lines.append("")

    # Matrix-vector reference
    lines.append("### Matrix-vector reference")
    if mv_actual == "dependency_skip":
        lines.append("NumPy unavailable – dependency_skip.")
    elif mv_actual == "expected_equal" and mv_obs.get("manual_result") is not None:
        mr = mv_obs.get("manual_result")
        nr = mv_obs.get("numpy_result")
        lines.append(f"manual_matvec result: {json.dumps(mr)}")
        lines.append(f"numpy.matmul result: {json.dumps(nr)}")
        lines.append(f"Classification: {mv_actual}")
    elif mv_actual == "fail":
        lines.append(f"Classification: fail – {mv_obs.get('reason','unknown')}")
    else:
        lines.append(f"Classification: {mv_actual}")
    lines.append("")

    # Shape contract
    lines.append("### Shape contract")
    if shape_actual == "dependency_skip":
        lines.append("NumPy unavailable – dependency_skip.")
    elif shape_actual == "expected_rejection":
        lines.append(f"manual_rejected={shape_obs.get('manual_rejected')}, numpy_rejected={shape_obs.get('numpy_rejected')}")
        lines.append(f"Classification: {shape_actual}")
    elif shape_actual == "fail":
        lines.append(f"Classification: fail – {shape_obs.get('reason','unknown')}")
    else:
        lines.append(f"Classification: {shape_actual}")
    lines.append("")

    # Stable softmax
    lines.append("### Stable softmax shift invariance")
    if softmax_actual == "dependency_skip":
        lines.append("NumPy unavailable – dependency_skip.")
    elif softmax_actual in ("expected_close", "fail", "pass"):
        out_logits = softmax_obs.get("output_logits")
        out_shifted = softmax_obs.get("output_shifted")
        if out_logits is not None:
            lines.append(f"output_logits: {json.dumps(out_logits)}")
        if out_shifted is not None:
            lines.append(f"output_shifted: {json.dumps(out_shifted)}")
        lines.append(f"finite={softmax_obs.get('finite')}, close_match={softmax_obs.get('close_match')}, input_unmutated={softmax_obs.get('input_unmutated')}")
        lines.append(f"Classification: {softmax_actual}")
    else:
        lines.append(f"Classification: {softmax_actual}")
    lines.append("")

    # Naive softmax overflow
    lines.append("### Naive softmax overflow")
    if naive_actual == "dependency_skip":
        lines.append("NumPy unavailable – dependency_skip.")
    else:
        lines.append(f"naive_has_nonfinite={naive_obs.get('naive_has_nonfinite')}, stable_all_finite={naive_obs.get('stable_all_finite')}, stable_sum_close_to_one={naive_obs.get('stable_sum_close_to_one')}")
        lines.append(f"Classification: {naive_actual}")
    lines.append("")

    # Floating comparison
    lines.append("### Floating comparison policy")
    if float_actual == "dependency_skip":
        lines.append("NumPy unavailable – dependency_skip.")
    else:
        lines.append(f"exact_equal={float_obs.get('exact_equal')}, close_ok={float_obs.get('close_ok')}")
        lines.append(f"Classification: {float_actual}")
    lines.append("")

    lines.append("## Evidence separation")
    lines.append("")
    lines.append("Hacker News commenters argued:")
    lines.append("- TrackerFF: professionals generally should not replace established numerical libraries with inferior hand-written implementations; questioned using elementary numerical routines as interview filters.")
    lines.append("- mchab: the project was intended as a learning tool inspired by learning through building (Karpathy zero-to-hero), rather than as interview preparation.")
    lines.append("- jvanderbot: implementing matrix-factorization routines helped them understand library design and performance decisions.")
    lines.append("- coliveira: writing an implementation does not by itself establish understanding of the underlying mathematics.")
    lines.append("- bArray: the exercises are a useful way to check understanding of fundamentals.")
    lines.append("- esafak: the difficult part of machine learning is often theory rather than implementation.")
    lines.append("- oehpr: an implementation exercise can provide motivation and a test of understanding.")
    lines.append("- drfunk: correctness problems around numerical comparisons; suggested numpy.testing.assert_allclose.")
    lines.append("- diimdeep: preferred ordinary local markdown and python files over a siloed exercise website.")
    lines.append("- ZoomerCretin: an example's matrix-vector notation was unclear.")
    lines.append("- r-zip: treated matrix multiplication as a basic machine-learning equivalent of fizzbuzz.")
    lines.append("")
    lines.append("Linked exercise site (https://www.deep-ml.com/): presents small ML coding exercises in a browser.")
    lines.append("")
    lines.append("Official NumPy documentation specifies:")
    lines.append("- numpy.matmul: matrix multiplication with shape-contract checking.")
    lines.append("- numpy.testing.assert_allclose: approximate floating-point comparison with rtol/atol.")
    lines.append("- numpy.errstate: control floating-point error handling.")
    lines.append("- numpy.exp: exponential function.")
    lines.append("- numpy.isfinite: finite-value checks.")
    lines.append("")
    lines.append("Local environment reported:")
    lines.append(f"- Python {platform.python_version()} ({platform.python_implementation()})")
    lines.append(f"- NumPy available={NUMPY_AVAILABLE}, version={NUMPY_VERSION}")
    lines.append("")
    lines.append("Lab directly observed:")
    # only list observations that actually passed
    if mv_actual == "expected_equal":
        lines.append("- manual_matvec agrees with numpy.matmul for the fixed 2×2 example.")
    if shape_actual == "expected_rejection":
        lines.append("- Both implementations reject the shape mismatch.")
    if softmax_actual == "expected_close":
        lines.append("- stable_softmax shift invariance holds approximately for the fixed inputs.")
    if naive_actual == "expected_nonfinite":
        lines.append("- Naive softmax overflows on [1000,1001,1002]; stable version is finite and sums ≈1.")
    if float_actual == "expected_close":
        lines.append("- Exact equality differs from approximate equality for [0.1+0.2, 1.0] vs [0.3, 1.0].")
    if not NUMPY_AVAILABLE:
        lines.append("- NumPy unavailable – numerical cases skipped.")
    lines.append("")
    lines.append("Lab did not test:")
    lines.append("- Other dtypes, batch shapes, sparse inputs, complex values, NaN/Inf handling, accelerators, or production workloads.")
    lines.append("- Statistical quality, model evaluation, training, or inference performance.")
    lines.append("- Interview prediction, learning effectiveness, or theoretical understanding.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("This repository does not prove:")
    lines.append("- that implementing numerical primitives is the best learning method for everyone;")
    lines.append("- that implementation ability establishes theoretical understanding;")
    lines.append("- that inability to implement a primitive from memory establishes lack of competence;")
    lines.append("- that this lab predicts interview or job performance;")
    lines.append("- that manual_matvec should replace numpy.matmul;")
    lines.append("- that stable_softmax is a complete production implementation;")
    lines.append("- that passing twenty-one rows proves general correctness;")
    lines.append("- that approximate equality is appropriate for every application;")
    lines.append("- that finite softmax values validate a model;")
    lines.append("- that the linked exercise site is correct or incorrect as a whole;")
    lines.append("- or that the lab is production-ready.")
    lines.append("")
    with open("RESULTS.md","w",encoding="utf-8",newline="\n") as f:
        f.write("\n".join(lines)+"\n")
    print(f"wrote {len(rows)} rows")
    print(dict(counts))
    return 0

if __name__ == "__main__":
    sys.exit(main())
