#!/usr/bin/env python3
import unittest
import json
import os
import re
import sys

with open("cases.json", "r", encoding="utf-8") as f:
    CASES_DOC = json.load(f)

CASE_IDS = CASES_DOC["case_ids"]
METHOD_IDS = CASES_DOC["method_ids"]
EXPECTED_MAP = CASES_DOC.get("expected_classifications", {})

import run_lab

class TestLab(unittest.TestCase):
    def test_case_ids(self):
        self.assertEqual(run_lab.CASE_IDS, [
            "runtime_and_dependency_marker",
            "matrix_vector_reference_marker",
            "matrix_vector_shape_contract_marker",
            "stable_softmax_shift_invariance_marker",
            "naive_softmax_overflow_marker",
            "floating_comparison_policy_marker",
            "no_learning_interview_or_production_validity_claim_marker",
        ])

    def test_method_ids(self):
        self.assertEqual(run_lab.METHOD_IDS, ["inspect_inputs", "execute_case", "verify_relation"])

    def test_twenty_one_rows(self):
        rows = run_lab.build_rows()
        self.assertEqual(len(rows), 21)
        seen = set()
        for r in rows:
            key = (r["case_id"], r["method_id"])
            self.assertNotIn(key, seen)
            seen.add(key)
        self.assertEqual(len(seen), 21)
        for cid in run_lab.CASE_IDS:
            for mid in run_lab.METHOD_IDS:
                self.assertIn((cid, mid), seen)

    def test_classification_vocabulary(self):
        valid = {"pass","expected_equal","expected_close","expected_rejection","expected_nonfinite","local_observation","context_only","dependency_skip","not_applicable","fail"}
        self.assertEqual(run_lab.VALID_CLASSIFICATIONS, valid)
        rows = run_lab.build_rows()
        for r in rows:
            self.assertIn(r["actual_classification"], valid)

    def test_expectation_map_coverage(self):
        for cid in CASE_IDS:
            for mid in METHOD_IDS:
                key = f"{cid}:{mid}"
                self.assertIn(key, EXPECTED_MAP, key)
                self.assertIn(EXPECTED_MAP[key], run_lab.VALID_CLASSIFICATIONS)

    def test_separate_expected_actual_fields(self):
        rows = run_lab.build_rows()
        for r in rows:
            self.assertIn("expected_classification", r)
            self.assertIn("actual_classification", r)

    def test_production_independence(self):
        # build with real expectation map
        rows1 = run_lab.build_rows()
        # build with a completely replaced expectation map
        # every expected_classification should be different from the original
        mutated = {}
        all_valid = list(run_lab.VALID_CLASSIFICATIONS)
        for k, v in EXPECTED_MAP.items():
            # pick a different valid classification
            for candidate in all_valid:
                if candidate != v:
                    mutated[k] = candidate
                    break
        rows2 = run_lab.build_rows(expected_map_override=mutated)
        # verify expected_classification fields changed
        changed_expected = False
        for a, b in zip(rows1, rows2):
            if a["expected_classification"] != b["expected_classification"]:
                changed_expected = True
                break
        self.assertTrue(changed_expected, "mutated expectation map had no effect on expected_classification field")
        # production outputs must remain unchanged
        for a, b in zip(rows1, rows2):
            self.assertEqual(a["case_id"], b["case_id"])
            self.assertEqual(a["method_id"], b["method_id"])
            self.assertEqual(a["actual_classification"], b["actual_classification"], f"actual classification changed for {a['case_id']}:{a['method_id']}")
            self.assertEqual(a["observation"], b["observation"], f"observation changed for {a['case_id']}:{a['method_id']}")

    def test_missing_handler_failure(self):
        orig = run_lab.HANDLERS.copy()
        try:
            del run_lab.HANDLERS["matrix_vector_reference_marker"]
            rows = run_lab.build_rows()
            fails = [r for r in rows if r["case_id"]=="matrix_vector_reference_marker" and r["actual_classification"]=="fail"]
            self.assertTrue(len(fails) >= 1)
            self.assertTrue(any("missing handler" in str(r["observation"].get("reason","")).lower() for r in fails))
        finally:
            run_lab.HANDLERS.clear()
            run_lab.HANDLERS.update(orig)

    def test_invalid_classification_failure(self):
        orig = run_lab.HANDLERS.get("floating_comparison_policy_marker")
        def bad_handler(method):
            return "not_a_valid_class", {}
        try:
            run_lab.HANDLERS["floating_comparison_policy_marker"] = bad_handler
            rows = run_lab.build_rows()
            fails = [r for r in rows if r["case_id"]=="floating_comparison_policy_marker" and r["actual_classification"]=="fail"]
            self.assertTrue(len(fails) >= 1)
            self.assertTrue(any("invalid classification" in str(r["observation"].get("reason","")).lower() for r in fails))
        finally:
            run_lab.HANDLERS["floating_comparison_policy_marker"] = orig

    def test_missing_classification_failure(self):
        # handler returns wrong shape / missing classification
        orig = run_lab.HANDLERS.get("floating_comparison_policy_marker")
        def bad_handler_no_class(method):
            # missing classification – return None / wrong tuple
            return (None, {})
        def bad_handler_wrong_tuple(method):
            return ["oops"]
        for bad in (bad_handler_no_class, bad_handler_wrong_tuple):
            try:
                run_lab.HANDLERS["floating_comparison_policy_marker"] = bad
                rows = run_lab.build_rows()
                fails = [r for r in rows if r["case_id"]=="floating_comparison_policy_marker" and r["actual_classification"]=="fail"]
                self.assertTrue(len(fails) >= 1)
                self.assertTrue(any("missing classification" in str(r["observation"].get("reason","")).lower() or "invalid classification" in str(r["observation"].get("reason","")).lower() for r in fails))
            finally:
                run_lab.HANDLERS["floating_comparison_policy_marker"] = orig

    def test_deterministic_row_ordering(self):
        r1 = run_lab.build_rows()
        r2 = run_lab.build_rows()
        self.assertEqual([(x["case_id"], x["method_id"]) for x in r1],
                         [(x["case_id"], x["method_id"]) for x in r2])

    def test_manual_matvec_exact_output(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        out = run_lab.manual_matvec([[1.0,2.0],[3.0,4.0]], [5.0,6.0])
        np.testing.assert_array_equal(out, np.array([17.0,39.0], dtype=np.float64))

    def test_manual_matvec_agrees_numpy(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        manual = run_lab.manual_matvec([[1.0,2.0],[3.0,4.0]], [5.0,6.0])
        numpy_result = np.matmul(np.array([[1.0,2.0],[3.0,4.0]]), np.array([5.0,6.0]))
        self.assertTrue(np.array_equal(manual, numpy_result))

    def test_shape_mismatch_rejection(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        with self.assertRaises(ValueError):
            run_lab.manual_matvec([[1,2,3],[4,5,6]], [7,8])
        with self.assertRaises(Exception):
            np.matmul(np.array([[1,2,3],[4,5,6]], dtype=np.float64), np.array([7,8], dtype=np.float64))

    def test_stable_softmax_finite(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        out = run_lab.stable_softmax([1000.0,1001.0,1002.0])
        self.assertTrue(np.all(np.isfinite(out)))

    def test_stable_softmax_sums_one(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        out = run_lab.stable_softmax([1000.0,1001.0,1002.0])
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=12)

    def test_stable_softmax_shift_invariance(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        o1 = run_lab.stable_softmax([1000.0,1001.0,1002.0])
        o2 = run_lab.stable_softmax([0.0,1.0,2.0])
        np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-12)

    def test_softmax_input_not_mutated(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        x = np.array([1000.0,1001.0,1002.0], dtype=np.float64)
        xcopy = x.copy()
        run_lab.stable_softmax(x)
        np.testing.assert_array_equal(x, xcopy)

    def test_naive_softmax_nonfinite(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        arr = np.array([1000.0,1001.0,1002.0], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            naive = np.exp(arr) / np.sum(np.exp(arr))
        self.assertFalse(bool(np.all(np.isfinite(naive))))

    def test_floating_exact_inequality(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        left = np.array([0.1+0.2, 1.0], dtype=np.float64)
        right = np.array([0.3, 1.0], dtype=np.float64)
        self.assertFalse(np.array_equal(left, right))

    def test_floating_assert_allclose(self):
        if not run_lab.NUMPY_AVAILABLE:
            self.skipTest("numpy not available")
        import numpy as np
        left = np.array([0.1+0.2, 1.0], dtype=np.float64)
        right = np.array([0.3, 1.0], dtype=np.float64)
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)

    def test_observations_json_agreement(self):
        with open("observations.json","r",encoding="utf-8") as f:
            disk = json.load(f)
        fresh = run_lab.build_rows()
        self.assertEqual(disk, fresh)

    def test_results_md_agreement(self):
        with open("RESULTS.md","r",encoding="utf-8") as f:
            text = f.read()
        rows = run_lab.build_rows()
        # row counts and classification totals
        self.assertIn(f"Rows: {len(rows)}", text)
        from collections import Counter
        counts = Counter(r["actual_classification"] for r in rows)
        for cls in run_lab.VALID_CLASSIFICATIONS:
            needle = f"- {cls}: {counts.get(cls,0)}"
            self.assertIn(needle, text, needle)
        # substantive observation agreement – build row map
        row_map = {(r["case_id"], r["method_id"]): r for r in rows}
        # matrix-vector
        mv = row_map.get(("matrix_vector_reference_marker", "verify_relation"))
        if mv and mv["actual_classification"] == "expected_equal":
            obs = mv["observation"]
            mr = obs.get("manual_result")
            nr = obs.get("numpy_result")
            if mr is not None:
                self.assertIn(json.dumps(mr), text, "manual_matvec result not in RESULTS.md")
            if nr is not None:
                self.assertIn(json.dumps(nr), text, "numpy.matmul result not in RESULTS.md")
        # shape contract
        shape = row_map.get(("matrix_vector_shape_contract_marker", "verify_relation"))
        if shape:
            obs = shape["observation"]
            if "manual_rejected" in obs:
                self.assertIn(str(obs["manual_rejected"]).lower(), text.lower())
            if "numpy_rejected" in obs:
                self.assertIn(str(obs["numpy_rejected"]).lower(), text.lower())
        # softmax shift invariance
        sm = row_map.get(("stable_softmax_shift_invariance_marker", "verify_relation"))
        if sm and sm["actual_classification"] == "expected_close":
            obs = sm["observation"]
            if obs.get("output_logits") is not None:
                self.assertIn(json.dumps(obs["output_logits"]), text)
        # naive softmax
        naive = row_map.get(("naive_softmax_overflow_marker", "verify_relation"))
        if naive:
            obs = naive["observation"]
            if "naive_has_nonfinite" in obs:
                self.assertIn(str(obs["naive_has_nonfinite"]).lower(), text.lower())
        # floating comparison
        flt = row_map.get(("floating_comparison_policy_marker", "verify_relation"))
        if flt:
            obs = flt["observation"]
            if "exact_equal" in obs:
                self.assertIn(str(obs["exact_equal"]).lower(), text.lower())
            if "close_ok" in obs:
                self.assertIn(str(obs["close_ok"]).lower(), text.lower())

    def test_required_non_claims(self):
        for path in ["README.md", "RESULTS.md"]:
            self.assertTrue(os.path.exists(path), f"{path} missing")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().lower()
            must_have = [
                "implementing numerical primitives is the best learning method",
                "implementation ability establishes theoretical understanding",
                "inability to implement",
                "predicts interview",
                "manual_matvec should replace numpy.matmul",
                "stable_softmax is a complete production implementation",
                "passing twenty-one rows proves general correctness",
                "approximate equality is appropriate for every application",
                "finite softmax values validate a model",
                "linked exercise site is correct or incorrect as a whole",
                "production-ready",
            ]
            for phrase in must_have:
                self.assertIn(phrase, text, f"missing non-claim phrase in {path}: {phrase}")

    # --- artifact scanner with explicit allowlist ---

    def _scan_patterns(self):
        """Return list of (compiled_regex, desc)."""
        return [
            (re.compile(r"/home/[^\s]+"), "home path"),
            (re.compile(r"/tmp/\S+"), "tmp path"),
            (re.compile(r"/clean-checkout"), "clean-checkout path"),
            (re.compile(r"[A-Za-z]:\\[^\s]+"), "windows absolute path"),
            (re.compile(r"ghp_[A-Za-z0-9]{36}"), "github pat"),
            (re.compile(r"github_pat_[A-Za-z0-9_]+"), "github fine-grained pat"),
            (re.compile(r"\bgho_[A-Za-z0-9]+"), "github oauth"),
            (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"), "bearer token"),
            (re.compile(r"AKIA[0-9A-Z]{16}"), "aws access key"),
            (re.compile(r"sk-live-[A-Za-z0-9]+"), "secret key"),
            (re.compile(r"api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]", re.I), "api key assignment"),
            (re.compile(r"password\s*[:=]\s*['\"][^'\"]{4,}['\"]", re.I), "password assignment"),
            (re.compile(r"token\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{20,}['\"]", re.I), "token assignment"),
            (re.compile(r"0x[0-9a-fA-F]{7,}"), "object address"),
            (re.compile(r"/[^\s]*openclaw[^\s]*", re.I), "workspace path"),
            (re.compile(r"/[^\s]*\.openclaw[^\s]*", re.I), "workspace path"),
            (re.compile(r"\bpid[=:]\s*\d+", re.I), "pid"),
            (re.compile(r"process[_\s]*id\s*[:=]\s*\d+", re.I), "process id"),
            (re.compile(r"\[\d{4,}\]"), "pid bracket"),
            (re.compile(r"hostname\s*[:=]\s*[^\s]+", re.I), "hostname"),
            (re.compile(r"\bhost\s*[:=]\s*[a-zA-Z0-9.-]+\b", re.I), "hostname"),
            (re.compile(r"\bUSER\s*=\s*\S+@\S+"), "user env dump"),
            (re.compile(r"\bHOME\s*=\s*/[^\s]+"), "home env dump"),
            (re.compile(r"\bPATH\s*=\s*[^\n]*:/home/", re.I), "path env dump"),
        ]

    def _scan_full_content_patterns(self):
        """Patterns that must scan full file content (multi-line)."""
        return [
            (re.compile(r"Traceback \(most recent call last\):.*?File \"[^\"]+\"", re.S), "traceback with path"),
        ]

    def _scan_file_content(self, path, content, allowlist=None):
        """Return list of (lineno, desc, match_text) violations found in content.
        allowlist is a set of (file, lineno, desc) tuples that are explicitly permitted.
        """
        if allowlist is None:
            allowlist = set()
        patterns = self._scan_patterns()
        full_content_patterns = self._scan_full_content_patterns()
        violations = []
        # full-content scan first (tracebacks)
        for pat, desc in full_content_patterns:
            m = pat.search(content)
            if m:
                # allowlist check – line 1 for full-content matches
                if (path, 1, desc) in allowlist:
                    continue
                violations.append((1, desc, m.group(0)[:80]))
        # line-by-line scan
        for lineno, line in enumerate(content.splitlines(), 1):
            for pat, desc in patterns:
                m = pat.search(line)
                if not m:
                    continue
                if (path, lineno, desc) in allowlist:
                    continue
                violations.append((lineno, desc, m.group(0)[:80]))
        return violations

    def test_artifact_scanner(self):
        required = ["README.md", "RESULTS.md", "cases.json", "observations.json", "run_lab.py", "test_lab.py", "hn_thread_evidence.md", "hn_comments_sanitized.json", ".gitignore"]
        if os.path.exists("VERIFY.md"):
            required.append("VERIFY.md")
        # explicit allowlist: (file, lineno, desc)
        # each entry permits one specific pattern match at an exact file location
        ALLOWLIST = {
            # test_lab.py – pattern definitions that self-match
            ("test_lab.py", 306, "home path"),
            ("test_lab.py", 307, "tmp path"),
            ("test_lab.py", 308, "clean-checkout path"),
            ("test_lab.py", 320, "workspace path"),
            ("test_lab.py", 321, "workspace path"),
            ("test_lab.py", 329, "home path"),
            # test_lab.py – negative test case strings in test_artifact_scanner_negative
            ("test_lab.py", 411, "home path"),
            ("test_lab.py", 412, "tmp path"),
            ("test_lab.py", 413, "github pat"),
            ("test_lab.py", 414, "aws access key"),
            ("test_lab.py", 415, "token assignment"),
            ("test_lab.py", 416, "password assignment"),
            ("test_lab.py", 417, "object address"),
            ("test_lab.py", 418, "home path"),
            ("test_lab.py", 418, "workspace path"),
            ("test_lab.py", 419, "pid"),
            ("test_lab.py", 420, "hostname"),
            ("test_lab.py", 435, "tmp path"),
            # .gitignore – intentional clean-checkout ignore
            (".gitignore", 12, "clean-checkout path"),
            # VERIFY.md – documented clone/checkout commands
            ("VERIFY.md", 10, "clean-checkout path"),
            ("VERIFY.md", 11, "clean-checkout path"),
            ("VERIFY.md", 12, "clean-checkout path"),
        }
        for path in required:
            self.assertTrue(os.path.exists(path), path)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            violations = self._scan_file_content(path, content, allowlist=ALLOWLIST)
            self.assertEqual(violations, [], f"{path} scanner violations: {violations}")

    def test_artifact_scanner_negative(self):
        """Verify scanner rejects prohibited content even with 'allow scanner pattern' marker."""
        # Test cases: (forbidden_text, expected_desc)
        test_cases = [
            ("/home/alice/secrets.txt", "home path"),
            ("/tmp/evil12345/payload", "tmp path"),
            ("ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "github pat"),
            ("AKIAIOSFODNN7EXAMPLE", "aws access key"),
            ('token = "abc123def456ghi789jkl0"', "token assignment"),
            ('password = "hunter2"', "password assignment"),
            ("0x7f8b1c2d3e4f", "object address"),
            ("/home/ubuntu/.openclaw/workspace/keys", "workspace path"),
            ("pid=12345", "pid"),
            ("hostname = buildbox-07", "hostname"),
        ]
        for forbidden, expected_desc in test_cases:
            # Test without marker – must be caught
            content = f"some text {forbidden} more text\n"
            violations = self._scan_file_content("README.md", content, allowlist=set())
            self.assertTrue(any(v[1] == expected_desc for v in violations),
                f"scanner failed to catch {expected_desc}: {forbidden!r}")
            # Test WITH generic bypass marker – must STILL be caught
            # (the old scanner had a generic "allow scanner pattern" bypass – that must NOT work)
            content_marked = f"some text {forbidden}  allow scanner pattern more\n"
            violations_marked = self._scan_file_content("README.md", content_marked, allowlist=set())
            self.assertTrue(any(v[1] == expected_desc for v in violations_marked),
                f"scanner incorrectly allowed {expected_desc} with generic bypass marker: {forbidden!r}")
        # Test traceback detection across multiple lines
        tb_content = "Traceback (most recent call last):\n  File \"/tmp/malicious.py\", line 1\n"
        tb_violations = self._scan_file_content("RESULTS.md", tb_content, allowlist=set())
        self.assertTrue(any(v[1] == "traceback with path" for v in tb_violations),
            "scanner failed to catch multi-line traceback")

if __name__ == "__main__":
    unittest.main()
