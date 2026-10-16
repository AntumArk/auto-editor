"""CPU-only report tests: python -m unittest discover -s presentation-pipeline/tests."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "pipeline_report", Path(__file__).resolve().parents[1] / "app" / "report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ChapterTests(unittest.TestCase):
    def test_order_spacing_and_titles(self):
        segments = [
            {"start": 0, "end": 8, "text": "Introduction"},
            {"start": 10, "end": 18, "text": "Second topic"},
            {"start": 25, "end": 30, "text": "Third topic"},
        ]
        chapters = report.make_chapters([25, 2, 10, 11, 10], segments, 45)
        self.assertEqual(chapters, [(0, "Introduction"), (10, "Second topic"), (25, "Third topic")])
        self.assertTrue(all(b[0] - a[0] >= 10 for a, b in zip(chapters, chapters[1:])))
        self.assertGreaterEqual(45 - chapters[-1][0], 10)

    def test_minimum_and_short_recordings(self):
        self.assertEqual(report.make_chapters([], [], 29.9), [])
        self.assertEqual([t for t, _ in report.make_chapters([], [], 30)], [0, 10, 20])
        self.assertEqual(len(report.make_chapters([15], [], 30)), 3)
        self.assertEqual(report.make_chapters([], [], 50, min_gap_s=20), [])
        self.assertEqual(len(report.make_chapters([], [], 30, min_gap_s=1)), 3)

    def test_word_titles_and_truncation(self):
        segments = [{"start": 0, "end": 30, "text": "Old topic New topic",
                     "words": [{"start": 0, "word": "Old"},
                               {"start": 10, "word": "New " + "topic " * 20}]}]
        chapters = report.make_chapters([10, 20], segments, 30)
        self.assertTrue(chapters[1][1].startswith("New"))
        self.assertLessEqual(len(chapters[1][1]), 60)

    def test_contiguous_segment_titles(self):
        segments = [
            {"start": 0, "end": 10, "text": "Opening remarks"},
            {"start": 10, "end": 20, "text": "Product architecture"},
            {"start": 20, "end": 30, "text": "Deployment results"},
        ]
        chapters = report.make_chapters([10, 20], segments, 30)
        self.assertEqual([title for _, title in chapters],
                         [segment["text"] for segment in segments])

    def test_report_format_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            report.write_report(
                path, "Talk", "Summary.", [(0, "Intro"), (60, "Topic"), (3600, "End")],
                [{"time": 65, "message": "Check this"}],
                {"original_duration": 4000, "edited_duration": 3700, "number_of_cuts": 4},
            )
            text = path.read_text(encoding="utf-8")
        for expected in ("00:00 Intro", "01:00 Topic", "1:00:00 End", "[01:05]",
                         "Original duration: 4000.00", "Edited duration: 3700.00",
                         "Time removed: 300.00", "Number of cuts: 4"):
            self.assertIn(expected, text)

    def test_omission_note(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            report.write_report(path, "Short", "", [], [], {})
            self.assertIn("Omitted:", path.read_text())


class ReviewTests(unittest.TestCase):
    def test_flags_and_edited_timestamps(self):
        transcript = [
            {"start": 0, "end": 2, "text": "Pradžia."},
            {"start": 20, "end": 25, "text": "Čia Kubernetes and the system.",
             "avg_logprob": -1.2,
             "words": [{"start": 21, "word": "Kubernetes", "probability": 0.2}]},
        ]
        notes = report.review_notes(
            transcript, [{"start": 20, "needs_review": True, "reason": "Unsure"}],
            [(3, 8), (6, 10)], {"transcription": {"language": "lt"}},
        )
        self.assertEqual([n["time"] for n in notes], sorted(n["time"] for n in notes))
        self.assertTrue(any(n["time"] == 13 and "Low-confidence" in n["message"] for n in notes))
        for flag in ("pause", "language switching", "rare names", "word probability", "LLM decision"):
            self.assertTrue(any(flag in n["message"] for n in notes), flag)

    def test_removed_pause_not_flagged(self):
        notes = report.review_notes(
            [{"start": 0, "end": 1, "text": "One."},
             {"start": 20, "end": 21, "text": "Two."}],
            [], {"cuts": [{"start": 1, "end": 20}]}, {},
        )
        self.assertEqual(notes, [])

    def test_summary_disabled_and_offline(self):
        transcript = {"segments": [{"text": "One. Two! Three? Four."}]}
        with patch.dict(sys.modules, {"requests": None}):
            self.assertEqual(report.summarize(transcript, {}), "One. Two! Three?")
            self.assertEqual(report.summarize(transcript, {"llm": {"enabled": True}}),
                             "One. Two! Three?")

    def test_summary_ollama_language_and_failure(self):
        requests = types.ModuleType("requests")
        requests.RequestException = RuntimeError
        response = unittest.mock.Mock()
        response.json.return_value = {"response": " Santrauka. "}
        requests.post = unittest.mock.Mock(return_value=response)
        cfg = {"llm": {"enabled": True}, "report": {"language": "lt"}}
        with patch.dict(sys.modules, {"requests": requests}):
            self.assertEqual(report.summarize([{"text": "Talk."}], cfg), "Santrauka.")
            self.assertIn("in lt", requests.post.call_args.kwargs["json"]["prompt"])
            requests.post.side_effect = RuntimeError("Offline")
            self.assertEqual(report.summarize([{"text": "Talk."}], cfg), "Talk.")


if __name__ == "__main__":
    unittest.main()
