"""CPU unit tests of the confirmatory guards (no model, no images, no attack).

Run from the project folder: python -m unittest tests.test_confirmatory -v
"""
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import confirmatory as c
import execution
import phase1b

CF = c.CF


def targets(n=852):
    return [{"id": i} for i in range(n)]


def manifest(**probe_overrides):
    probe = {"n_full": 852, "estimate_hours": 4.0, "probe_target_ids": [785, 2153], "warm_up": True}
    probe.update(probe_overrides)
    probe = {k: v for k, v in probe.items() if v is not DROP}
    return {"probe": probe, "target_range": [CF["first_target"], CF["first_target"] - 1 + 852]}


DROP = object()


class TestSelectRange(unittest.TestCase):
    def test_valid_manifest_selects_full_range(self):
        self.assertEqual(len(c.select_range(targets(), manifest())), 852)

    def test_missing_warm_up_rejected(self):
        with self.assertRaises(SystemExit) as e:
            c.select_range(targets(), manifest(warm_up=DROP))
        self.assertIn("warm_up", str(e.exception))

    def test_warm_up_false_rejected(self):
        with self.assertRaises(SystemExit) as e:
            c.select_range(targets(), manifest(warm_up=False))
        self.assertIn("warm_up", str(e.exception))

    def test_warm_up_truthy_but_not_true_rejected(self):
        with self.assertRaises(SystemExit):
            c.select_range(targets(), manifest(warm_up=1))

    def test_one_probe_target_rejected(self):
        with self.assertRaises(SystemExit):
            c.select_range(targets(), manifest(probe_target_ids=[785]))

    def test_range_must_follow_from_estimate(self):
        with self.assertRaises(SystemExit):
            c.select_range(targets(), manifest(estimate_hours=CF["runtime_limit_hours"] + 1))

    def test_fallback_boundary(self):
        limit = CF["runtime_limit_hours"]
        self.assertEqual(c.choose_range(targets(), limit)[1], CF["first_target"] - 1 + 852)
        self.assertEqual(c.choose_range(targets(), limit + 0.01)[1], CF["fallback_last_target"])


class TestProvenanceOnEmptyOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "manifest.json"
        path.write_text("{}", encoding="utf-8")
        self.prov = c.provenance(path, test=True)

    def tearDown(self):
        self.tmp.cleanup()

    def roundtrip(self, df):
        buf = io.StringIO()
        c.with_provenance(df, self.prov).to_csv(buf, index=False)
        return buf.getvalue(), pd.read_csv(io.StringIO(buf.getvalue()))

    def assert_has_provenance(self, text):
        self.assertIn(c.TEST_LABEL, text)
        self.assertIn(self.prov["manifest_path"], text)
        self.assertIn(self.prov["manifest_sha256"], text)

    def test_empty_targets_table(self):
        text, back = self.roundtrip(phase1b.table([]))
        self.assert_has_provenance(text)
        self.assertEqual(len(back), 1)
        self.assertEqual(back.loc[0, "row_type"], c.METADATA_ROW)
        self.assertTrue(set(phase1b.TABLE_COLUMNS) <= set(back.columns))

    def test_empty_selection_report(self):
        text, back = self.roundtrip(c.selection_report([]))
        self.assert_has_provenance(text)
        self.assertEqual(back.loc[0, "row_type"], c.METADATA_ROW)

    def test_zero_eligible_outputs(self):
        sel = [{"id": i, "eligible": False, "conditions": {k: {"clean_conf": None} for k in c.CONDS}}
               for i in (1, 2)]
        df, summary, control, dec, outcome = c.analyze(sel)
        self.assertEqual(outcome, "does not support either direction")
        for table in (df, summary, control, dec, c.selection_report(sel)):
            self.assert_has_provenance(self.roundtrip(table)[0])
        self.assertIn(c.NO_ELIGIBLE, summary["value"].astype(str).tolist())

    def test_non_empty_wide_table_unchanged_apart_from_columns(self):
        df = pd.DataFrame({"image_id": [1, 2]})
        _, back = self.roundtrip(df)
        self.assertEqual(len(back), 2)
        self.assertNotIn("row_type", back.columns)
        self.assertTrue((back["run_label"] == c.TEST_LABEL).all())


class TestProbeEligibility(unittest.TestCase):
    PILOT = [{"id": i, "file_name": f"{i}.jpg", "target_bbox": [i, 0, 1, 1]} for i in (785, 2153, 2261)]

    def run_probe(self, eligible_ids):
        def clean(predictor, src, bbox, f):
            ok = bbox[0] in eligible_ids
            return ({k: {"detected": ok, "gt": [0, 0, 1, 1]} for k in c.CONDS}, {k: None for k in c.CONDS})

        m1 = {"pgd_seconds_per_step": 1.0, "detect_seconds_per_step": 0.5}
        m2 = {"tries": [{"pgd_seconds": 20.0}]}
        with mock.patch.object(c.data, "ensure_images"), mock.patch.object(c.cv2, "imread"), \
                mock.patch.object(c.phase1, "clean_conditions", clean), \
                mock.patch.object(c, "warm_up") as warm, \
                mock.patch.object(c.phase1b, "measure", return_value=(m1, m2)):
            return c.probe_timing(self.PILOT, 1.0, lambda: (None, None)), warm

    def test_zero_eligible_aborts(self):
        with self.assertRaises(SystemExit):
            self.run_probe(set())

    def test_one_eligible_aborts(self):
        with self.assertRaises(SystemExit):
            self.run_probe({785})

    def test_skips_ineligible_until_two(self):
        timing, warm = self.run_probe({785, 2261})
        self.assertEqual(timing["probe_target_ids"], [785, 2261])
        self.assertIs(timing["warm_up"], True)
        warm.assert_called_once()


class TestProductionProtocol(unittest.TestCase):
    def good(self):
        return {"device": "cuda:0", "gpu_count": 1, "gpu_name": "Tesla T4", "dtype": "float32",
                "flags": dict(execution.EXPECTED_FLAGS), "env": dict(execution.CUDA_ENV),
                **execution.pinned_versions()}

    def test_good_record_passes(self):
        execution.validate_production(self.good())

    def test_cpu_record_rejected(self):
        bad = {**self.good(), "device": "cpu", "gpu_count": 0, "gpu_name": None}
        with self.assertRaises(SystemExit):
            execution.validate_production(bad)

    def test_t4_names_accepted(self):
        for name in ("Tesla T4", "NVIDIA T4", "  Tesla T4\n"):
            execution.validate_production({**self.good(), "gpu_name": name})

    def test_other_gpu_names_rejected(self):
        for name in ("NVIDIA T400", "T4000", "Tesla T40", "T4", "tesla t4", "NVIDIA Tesla T4",
                     "Tesla T4 (2)", "", None):
            with self.subTest(name=name), self.assertRaises(SystemExit):
                execution.validate_production({**self.good(), "gpu_name": name})

    def test_flag_and_version_rejected(self):
        for change in ({"flags": {**execution.EXPECTED_FLAGS, "cudnn_benchmark": True}},
                       {"scipy": "0.0.0"}, {"dtype": "float16"}):
            with self.assertRaises(SystemExit):
                execution.validate_production({**self.good(), **change})


class TestHolm(unittest.TestCase):
    def test_holm(self):
        self.assertEqual(c.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        adj = c.holm([0.5, math.nan, 0.01])
        self.assertEqual(adj[0], 1.0)
        self.assertTrue(math.isnan(adj[1]))
        self.assertEqual(adj[2], 0.03)


if __name__ == "__main__":
    unittest.main()
