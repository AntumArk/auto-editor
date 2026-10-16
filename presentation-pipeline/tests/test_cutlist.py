import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
from cutlist import CutList, build_cutlist, export_with_auto_editor
from vision import detect_slide_changes


def config(**cuts):
    return SimpleNamespace(cuts=SimpleNamespace(
        **{"target_pause_s": 0.6, "full_remove_min_s": 8, **cuts},
    ))


def candidate(start=2, end=6, id="pause"):
    return SimpleNamespace(id=id, start=start, end=end, duration=end - start, context={})


class CutListTests(unittest.TestCase):
    def test_default_shortens_center_and_preserves_edges(self):
        result = build_cutlist([candidate()], {}, 10, config())
        self.assertEqual(result.removed, [(2.3, 5.7)])
        self.assertEqual(result.keep, [(0, 2.3), (5.7, 10)])
        self.assertAlmostEqual(result.original_duration, 10)
        self.assertAlmostEqual(result.edited_duration, 6.6)

    def test_keep_and_unknown_candidate_decisions(self):
        for decisions in ({"pause": {"action": "keep"}},
                          {"invented": {"action": "approve", "start": 0, "end": 10}}):
            with self.subTest(decisions=decisions):
                result = build_cutlist([candidate()], decisions, 10, config())
                self.assertEqual(bool(result.removed), "pause" not in decisions)

    def test_decisions_list_accepts_candidate_id_or_id_on_dicts_and_objects(self):
        for key in ("candidate_id", "id"):
            for decision in ({"action": "keep", key: "pause"},
                             SimpleNamespace(action="keep", **{key: "pause"})):
                with self.subTest(decision=decision):
                    result = build_cutlist([candidate()], [decision], 10, config())
                    self.assertEqual(result.removed, [])
                    self.assertEqual(result.keep, [(0, 10)])

    def test_default_full_removal_threshold_matches_config(self):
        result = build_cutlist(
            [candidate(end=10)], {"pause": {"action": "approve"}}, 12, {},
        )
        self.assertEqual(result.removed, [(2, 10)])

    def test_cutlist_json_round_trip_restores_tuple_intervals(self):
        result = build_cutlist([candidate()], {}, 10, config())
        restored = CutList.from_dict(json.loads(result.to_json()))
        self.assertEqual(restored, result)
        self.assertIsInstance(restored.keep[0], tuple)
        self.assertIsInstance(restored.removed[0], tuple)

    def test_full_removal_requires_approval_and_minimum_length(self):
        for action, end, expected in (
            ("approve", 10, [(2.0, 10.0)]),
            ("approve", 6, [(2.3, 5.7)]),
            ("shorten", 10, [(2.3, 9.7)]),
        ):
            with self.subTest(action=action, end=end):
                result = build_cutlist(
                    [candidate(end=end)], {"pause": {"action": action}}, 12, config(),
                )
                self.assertEqual(result.removed, expected)

    def test_positive_override_keeps_requested_pause_even_when_approved(self):
        result = build_cutlist(
            [candidate(end=10)], {"pause": {"action": "approve", "target_pause_s": 2}},
            12, config(),
        )
        self.assertEqual(result.removed, [(3, 9)])

    def test_overrides_are_clamped_and_cannot_bypass_full_removal_gate(self):
        for action, target, end, expected in (
            ("shorten", 20, 6, []),
            ("shorten", 0, 10, []),
            ("approve", 0, 6, []),
            ("approve", 0, 10, [(2, 10)]),
        ):
            with self.subTest(action=action, target=target, end=end):
                result = build_cutlist(
                    [candidate(end=end)],
                    {"pause": {"action": action, "target_pause_s": target}},
                    12, config(),
                )
                self.assertEqual(result.removed, expected)

    def test_invalid_decisions_fall_back_to_shortening(self):
        for decision in (None, "approve", {"action": "delete"},
                         {"action": "shorten", "target_pause_s": -1},
                         {"action": "approve", "target_pause_s": math.nan},
                         {"action": "shorten", "target_pause_s": "bad"}):
            with self.subTest(decision=decision):
                result = build_cutlist([candidate()], {"pause": decision}, 10, config())
                self.assertEqual(result.removed, [(2.3, 5.7)])

    def test_empty_candidates_and_zero_duration(self):
        self.assertEqual(build_cutlist([], None, 10, config()).keep, [(0, 10)])
        result = build_cutlist([], {}, 0, config())
        self.assertEqual(result.keep, [])
        self.assertEqual(result.edited_duration, 0)

    def test_unsorted_overlaps_merge_and_partition_source(self):
        result = build_cutlist(
            [candidate(5, 9, "b"), candidate(2, 7, "a")], {}, 12, config(),
        )
        self.assertEqual(result.removed, [(2.3, 8.7)])
        spans = sorted(result.removed + result.keep)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 12)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))
        self.assertAlmostEqual(result.edited_duration, 5.6)

    def test_invalid_bounds_and_config_rejected(self):
        for start, end in ((-1, 4), (4, 3), (2, 11), (0, math.inf), (math.nan, 4)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                build_cutlist([candidate(start, end)], {}, 10, config())
        for duration in (-1, math.nan, math.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                build_cutlist([], {}, duration, config())
        with self.assertRaises(ValueError):
            build_cutlist([], {}, 10, config(target_pause_s=-1))

    def test_json_and_mapping_config(self):
        result = build_cutlist([candidate()], {}, 10, {"cuts": {"target_pause_s": 2}})
        self.assertEqual(result.removed, [(3, 5)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cutlist.json"
            self.assertEqual(json.loads(result.to_json(path)), json.loads(path.read_text()))
        self.assertEqual(json.loads(result.to_json())["removed"], [[3, 5]])


class ExportTests(unittest.TestCase):
    @patch("cutlist.subprocess.run")
    def test_original_stem_names_all_exports(self, run):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = export_with_auto_editor(
                Path(directory) / "source.mp4", build_cutlist([], {}, 10, {}),
                directory, {"export": {"preview": True}}, name="Original talk",
            )
            self.assertEqual(
                {path.name for path in artifacts.values()},
                {"Original talk.cutlist.json", "Original talk.fcpxml", "Original talk.preview.mp4"},
            )
            self.assertEqual(
                [Path(call.args[0][-1]).name for call in run.call_args_list],
                ["Original talk.fcpxml", "Original talk.preview.mp4"],
            )

    @patch("cutlist.subprocess.run")
    def test_resolve_and_preview_use_same_explicit_second_cuts(self, run):
        result = build_cutlist([candidate()], {}, 10, config())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "-talk ; quoted.mp4"
            cfg = {"export": {"preview": True}}
            artifacts = export_with_auto_editor(source, result, directory, cfg)
            self.assertEqual(json.loads(artifacts["cutlist"].read_text())["removed"], [[2.3, 5.7]])
            self.assertEqual(artifacts["resolve"].suffix, ".fcpxml")
            self.assertEqual(artifacts["preview"].suffix, ".mp4")
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[:4], ["auto-editor", str(source), "--edit", "all"])
                self.assertEqual(command[4:6], ["--cut-out", "2.300000000sec,5.700000000sec"])
                self.assertEqual(call.kwargs, {"check": True})
            self.assertEqual(run.call_args_list[0].args[0][6:8], ["--export", "resolve"])
            self.assertEqual(run.call_args_list[1].args[0][6:8], ["--export", "default"])

    @patch("cutlist.subprocess.run")
    def test_no_cuts_omits_cut_flag_and_preserves_whole_source(self, run):
        with tempfile.TemporaryDirectory() as directory:
            export_with_auto_editor("talk.mp4", build_cutlist([], {}, 10, {}), directory, {})
        command = run.call_args.args[0]
        self.assertNotIn("--cut-out", command)
        self.assertEqual(command[2:4], ["--edit", "all"])
        run.assert_called_once()

    @patch("cutlist.subprocess.run", side_effect=subprocess.CalledProcessError(1, "auto-editor"))
    def test_export_failure_propagates(self, run):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(subprocess.CalledProcessError):
            export_with_auto_editor("talk.mp4", CutList([], [(0, 10)], 10, 10), directory, {})


class VisionTests(unittest.TestCase):
    def fake_cv2(self, differences, fractions, frames=5, fps=2):
        cv2 = MagicMock()
        capture = cv2.VideoCapture.return_value
        capture.get.side_effect = [fps, frames]
        capture.grab.side_effect = [True] * frames + [False]
        frame = MagicMock()
        frame.shape = (4, 8, 3)
        capture.retrieve.return_value = (True, frame)
        deltas = []
        for difference in differences:
            delta = MagicMock()
            delta.mean.return_value = difference * 255
            deltas.append(delta)
        cv2.absdiff.side_effect = deltas
        mask = MagicMock()
        mask.size = 1000
        cv2.threshold.return_value = (None, mask)
        cv2.countNonZero.side_effect = [fraction * 1000 for fraction in fractions]
        return cv2, capture

    def test_transition_sampling_window_excluded_and_cursor_noise_ignored(self):
        cv2, capture = self.fake_cv2([0.05, 0.2, 0, 0], [0.005, 0.5, 0, 0])
        with patch.dict(sys.modules, {"cv2": cv2}):
            result = detect_slide_changes("talk.mp4", {
                "visual": {"sample_fps": 2, "downscale_width": 2},
            })
        self.assertEqual(result["change_times"], [1])
        self.assertEqual(result["static_intervals"], [(0, 0.5), (1, 2.5)])
        self.assertEqual(len(result["samples"]), 5)
        self.assertAlmostEqual(result["samples"][1][1], 0.05)
        cv2.resize.assert_called()
        capture.release.assert_called_once()

    def test_static_video_and_sampling_rate(self):
        cv2, capture = self.fake_cv2([0, 0], [0, 0])
        with patch.dict(sys.modules, {"cv2": cv2}):
            result = detect_slide_changes("talk.mp4", {"visual": {"sample_fps": 1}})
        self.assertEqual(result["static_intervals"], [(0, 2.5)])
        self.assertEqual(result["samples"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(capture.retrieve.call_count, 3)

    def test_invalid_settings_and_decode_errors(self):
        for visual in ({"sample_fps": 0}, {"sample_fps": math.inf},
                       {"downscale_width": 0}, {"change_threshold": math.nan}):
            with self.subTest(visual=visual), self.assertRaises(ValueError):
                detect_slide_changes("talk.mp4", {"visual": visual})
        cv2, capture = self.fake_cv2([], [], frames=1)
        capture.retrieve.return_value = (False, None)
        with patch.dict(sys.modules, {"cv2": cv2}), self.assertRaises(ValueError):
            detect_slide_changes("talk.mp4", {})
        capture.release.assert_called_once()

    def test_truncated_video_does_not_become_static(self):
        cv2, capture = self.fake_cv2([], [], frames=20)
        capture.grab.side_effect = [True, False]
        with patch.dict(sys.modules, {"cv2": cv2}), self.assertRaises(ValueError):
            detect_slide_changes("talk.mp4", {})
        capture.release.assert_called_once()

    def test_module_import_does_not_require_opencv(self):
        spec = importlib.util.spec_from_file_location("isolated_vision", APP / "vision.py")
        with patch.dict(sys.modules, {"cv2": None}):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))


if __name__ == "__main__":
    unittest.main()
