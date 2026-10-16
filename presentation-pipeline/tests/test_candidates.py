"""Dependency-free tests for the candidate-generation API."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SPEC = importlib.util.spec_from_file_location(
    "presentation_candidates", Path(__file__).parents[1] / "app" / "candidates.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Candidate = MODULE.Candidate
generate_candidates = MODULE.generate_candidates


class CandidateTests(unittest.TestCase):
    def generate(self, transcript=None, slide_info=None, duration=20, **settings):
        return generate_candidates(
            [] if transcript is None else transcript,
            {"static_intervals": [[0, duration]]} if slide_info is None else slide_info,
            duration,
            {"cuts": settings},
        )

    def bounds(self, candidates):
        return [(item.start, item.end) for item in candidates]

    def test_silence_only_in_static_intervals(self):
        result = self.generate(
            slide_info={"static_intervals": [[1, 4], [8, 9], [12, 18]]}
        )
        self.assertEqual(self.bounds(result), [(1, 4), (12, 18)])
        self.assertEqual([item.duration for item in result], [3, 6])
        self.assertTrue(all(isinstance(item, Candidate) for item in result))

    def test_segment_and_word_bounds_both_protected(self):
        result = self.generate(
            {
                "segments": [
                    {
                        "start": 3,
                        "end": 7,
                        "text": "whole segment",
                        "words": [{"start": 4, "end": 5, "word": "inside"}],
                    }
                ],
                "words": [{"start": 10, "end": 11, "word": "outside"}],
            },
            speech_margin_s=0,
        )
        self.assertEqual(self.bounds(result), [(0, 3), (7, 10), (11, 20)])
        self.assertEqual(result[1].context["transcript_before"], "whole segment")
        self.assertEqual(result[1].context["transcript_after"], "outside")

    def test_speech_margins_and_list_segments(self):
        result = self.generate(
            [{"start": 5, "end": 10, "text": "speech"}], speech_margin_s=1
        )
        self.assertEqual(self.bounds(result), [(0, 4), (11, 20)])

    def test_word_bounds_outside_segments_and_words_only(self):
        result = self.generate(
            {
                "segments": [
                    {
                        "start": 5,
                        "end": 8,
                        "words": [{"start": 3, "end": 10, "word": "wide"}],
                    }
                ]
            },
            speech_margin_s=0,
        )
        self.assertEqual(self.bounds(result), [(0, 3), (10, 20)])
        words_only = self.generate(
            {"words": [{"start": 3, "end": 10, "word": "wide"}]},
            speech_margin_s=0,
        )
        self.assertEqual(self.bounds(words_only), [(0, 3), (10, 20)])

    def test_slide_settling_and_context(self):
        result = self.generate(
            slide_info={"static_intervals": [[0, 20]], "change_times": [5, 12]},
            settle_after_slide_change_s=2,
        )
        self.assertEqual(self.bounds(result), [(0, 5), (7, 12), (14, 20)])
        self.assertEqual(result[1].context["previous_slide_change"], 5)
        self.assertEqual(result[1].context["next_slide_change"], 12)

    def test_zero_settling_never_bridges_or_starts_on_slide_change(self):
        result = self.generate(
            slide_info={
                "static_intervals": [[0, 10], [8, 20]],
                "change_times": [10],
            },
            settle_after_slide_change_s=0,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].end, 10)
        self.assertGreater(result[1].start, 10)
        self.assertFalse(any(item.start <= 10 < item.end for item in result))

    def test_change_times_is_canonical_even_when_empty(self):
        for changes, expected in (
            ([5], [(0, 5), (6, 20)]),
            ([], [(0, 20)]),
        ):
            with self.subTest(changes=changes):
                result = self.generate(
                    slide_info={
                        "static_intervals": [[0, 20]],
                        "change_times": changes,
                        "slide_changes": [12],
                    }
                )
                self.assertEqual(self.bounds(result), expected)

    def test_overlapping_static_and_protected_spans(self):
        result = self.generate(
            [{"start": 4, "end": 8}, {"start": 7, "end": 11}],
            {"static_intervals": [[0, 8], [6, 18], [18, 20]]},
            speech_margin_s=0,
        )
        self.assertEqual(self.bounds(result), [(0, 4), (11, 20)])

    def test_clamps_valid_bounds_and_settling_before_zero(self):
        result = self.generate(
            [{"start": -5, "end": 1}, {"start": 18, "end": 30}],
            {
                "static_intervals": [[-10, 40]],
                "change_times": [-1],
            },
            speech_margin_s=0,
            settle_after_slide_change_s=4,
        )
        self.assertEqual(self.bounds(result), [(3, 18)])

    def test_defaults_and_namespace_configuration(self):
        cfg = SimpleNamespace(cuts=SimpleNamespace())
        result = generate_candidates(
            {"segments": [{"start": 5, "end": 10}]},
            SimpleNamespace(static_intervals=[SimpleNamespace(start=0, end=20)]),
            20,
            cfg,
        )
        self.assertEqual(self.bounds(result), [(0, 4.8), (10.2, 20)])
        settled = self.generate(
            slide_info={"static_intervals": [[0, 20]], "changes": [5]}
        )
        self.assertEqual(self.bounds(settled), [(0, 5), (6, 20)])

    def test_empty_and_fully_protected_inputs(self):
        self.assertEqual(self.generate(slide_info={}), [])
        self.assertEqual(self.generate(duration=0), [])
        self.assertEqual(self.generate([{"start": 0, "end": 20}]), [])
        self.assertEqual(self.bounds(self.generate({})), [(0, 20)])

    def test_minimum_is_inclusive(self):
        self.assertEqual(
            self.bounds(self.generate(slide_info={"static_intervals": [[1, 3]]})),
            [(1, 3)],
        )
        self.assertEqual(
            self.generate(slide_info={"static_intervals": [[1, 2.99]]}), []
        )

    def test_ids_are_stable_unique_strings_and_inputs_unchanged(self):
        transcript = {"segments": [{"start": 4, "end": 8, "text": "hello"}]}
        slides = {"static_intervals": [[0, 20]], "change_times": [12, 12]}
        first = self.generate(transcript, slides)
        self.assertEqual(first, self.generate(transcript, slides))
        ids = [item.id for item in first]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(isinstance(item, str) for item in ids))
        self.assertEqual(slides["change_times"], [12, 12])
        self.assertEqual(transcript["segments"][0]["start"], 4)

    def test_invalid_transcript_bounds_raise_even_inside_valid_segment(self):
        bad_bounds = [
            {},
            {"start": 3},
            {"start": 4, "end": 3},
            {"start": float("nan"), "end": 8},
            {"start": 3, "end": float("inf")},
            {"start": "3", "end": 8},
            {"start": True, "end": 8},
        ]
        for bounds in bad_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    self.generate([bounds])
                with self.assertRaises(ValueError):
                    self.generate(
                        {"segments": [{"start": 0, "end": 20, "words": [bounds]}]}
                    )
                with self.assertRaises(ValueError):
                    self.generate({"words": [bounds]})

    def test_invalid_inputs_raise(self):
        for transcript in (
            None,
            "speech",
            {"segments": "invalid"},
            {"segments": None},
            {"words": None},
            {"segments": [{"start": 1, "end": 2, "words": None}]},
            {"text": "x"},
        ):
            with self.subTest(transcript=transcript):
                with self.assertRaises(ValueError):
                    generate_candidates(transcript, {}, 20, {})
        for duration in (-1, float("inf"), float("nan"), "20"):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    self.generate(duration=duration)
        for name in (
            "min_silence_s", "speech_margin_s", "settle_after_slide_change_s"
        ):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        self.generate(**{name: value})
        for slide_info in (
            {"static_intervals": [[5, 4]]},
            {"static_intervals": [[0, float("inf")]]},
            {"static_intervals": [[0, 20]], "change_times": [float("nan")]},
        ):
            with self.subTest(slide_info=slide_info):
                with self.assertRaises(ValueError):
                    self.generate(slide_info=slide_info)


if __name__ == "__main__":
    unittest.main()
