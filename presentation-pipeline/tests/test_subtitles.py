"""Run with: python -m unittest discover -s presentation-pipeline/tests."""

from copy import deepcopy
from decimal import Decimal
import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PIPELINE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_PATH))
try:
    subtitles = importlib.import_module("app.subtitles")
finally:
    sys.path.pop(0)


class MapTimeTests(unittest.TestCase):
    def test_namespace_package_import(self):
        self.assertEqual(subtitles.__name__, "app.subtitles")
        self.assertEqual(Path(subtitles.__file__).resolve(),
                         PIPELINE_PATH / "app" / "subtitles.py")

    def test_boundaries_gaps_and_clamping(self):
        keeps = [(2, 5), (8, 10), (15, 20)]
        for source, expected in [
            (-10, 0), (0, 0), (2, 0), (3.25, 1.25), (5, 3),
            (6, 3), (8, 3), (9, 4), (10, 5), (14, 5),
            (15, 5), (19, 9), (20, 10), (100, 10),
        ]:
            with self.subTest(source=source):
                self.assertEqual(subtitles.map_time(source, keeps), expected)

    def test_empty_and_zero_length_keeps(self):
        for keeps in [[], [(2, 2)]]:
            for source in [-1, 0, 2, 100]:
                self.assertEqual(subtitles.map_time(source, keeps), 0)

    def test_unsorted_overlapping_and_touching_keeps_form_union(self):
        keeps = [(8, 10), (4, 6), (2, 5), (6, 8), (2, 5)]
        original = deepcopy(keeps)
        self.assertEqual(subtitles.map_time(9, keeps), 7)
        self.assertEqual(subtitles.map_time(20, keeps), 8)
        self.assertEqual(keeps, original)

    def test_fractional_negative_and_iterable_intervals(self):
        self.assertAlmostEqual(
            subtitles.map_time(0.3, iter([(-0.2, 0.1), (0.2, 0.4)])), 0.4
        )

    def test_invalid_times_and_intervals(self):
        for invalid in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    subtitles.map_time(invalid, [])
                with self.assertRaises(ValueError):
                    subtitles.map_time(0, [(0, invalid)])
        with self.assertRaises(ValueError):
            subtitles.map_time(1, [(5, 2)])


class RemapSegmentsTests(unittest.TestCase):
    def test_partial_clipping_shifting_dropping_and_multiple_cuts(self):
        segments = [
            {"start": 0, "end": 3, "text": "left clipped"},
            {"start": 3, "end": 6, "text": "right clipped"},
            {"start": 5, "end": 8, "text": "removed"},
            {"start": 9, "end": 12, "text": "later"},
            {"start": 1, "end": 19, "text": "across cuts"},
            {"start": 21, "end": 23, "text": "after keeps"},
        ]
        mapped = subtitles.remap_segments(segments, [(2, 5), (8, 10), (15, 20)])
        self.assertEqual(
            [(item["start"], item["end"], item["text"]) for item in mapped],
            [(0, 1, "left clipped"), (1, 3, "right clipped"),
             (4, 5, "later"), (0, 9, "across cuts")],
        )

    def test_exact_boundaries_and_zero_length_segments(self):
        segments = [
            {"start": 0, "end": 2, "text": "touch left"},
            {"start": 5, "end": 8, "text": "gap"},
            {"start": 10, "end": 11, "text": "touch right"},
            {"start": 3, "end": 3, "text": "zero"},
            {"start": 2, "end": 5, "text": "exact first keep"},
            {"start": 8, "end": 10, "text": "exact second keep"},
        ]
        self.assertEqual(
            subtitles.remap_segments(segments, [(2, 5), (8, 10)]),
            [{"start": 0, "end": 3, "text": "exact first keep"},
             {"start": 3, "end": 5, "text": "exact second keep"}],
        )

    def test_words_clip_drop_shift_and_span_multiple_cuts(self):
        segments = [{
            "start": 0, "end": 20, "text": "all words",
            "words": [
                {"start": 0, "end": 3, "word": "left", "score": 0.9},
                {"start": 4, "end": 7, "word": "right"},
                {"start": 5, "end": 8, "word": "removed"},
                {"start": 8, "end": 8, "word": "zero"},
                {"start": 9, "end": 10, "word": "later"},
                {"start": 3, "end": 17, "word": "multiple cuts"},
                {"start": 18, "end": 25, "word": "last"},
            ],
        }]
        mapped = subtitles.remap_segments(segments, [(2, 5), (8, 10), (15, 20)])
        self.assertEqual((mapped[0]["start"], mapped[0]["end"]), (0, 10))
        self.assertEqual(mapped[0]["words"], [
            {"start": 0, "end": 1, "word": "left", "score": 0.9},
            {"start": 2, "end": 3, "word": "right"},
            {"start": 4, "end": 5, "word": "later"},
            {"start": 1, "end": 7, "word": "multiple cuts"},
            {"start": 8, "end": 10, "word": "last"},
        ])

    def test_nested_metadata_is_preserved_without_aliasing_or_mutation(self):
        segments = [{
            "start": 1, "end": 4, "text": "original",
            "metadata": {"tags": ["segment"]},
            "words": [{"start": 1, "end": 3, "word": "original",
                       "metadata": {"tags": ["word"]}}],
        }]
        keeps = [[2, 5]]
        original, original_keeps = deepcopy(segments), deepcopy(keeps)
        mapped = subtitles.remap_segments(segments, keeps)
        self.assertEqual(segments, original)
        self.assertEqual(keeps, original_keeps)
        self.assertEqual(mapped[0]["metadata"], segments[0]["metadata"])
        self.assertEqual(mapped[0]["words"][0]["metadata"],
                         segments[0]["words"][0]["metadata"])
        mapped[0]["metadata"]["tags"].append("changed")
        mapped[0]["words"][0]["metadata"]["tags"].append("changed")
        self.assertEqual(segments, original)

    def test_empty_inputs_and_optional_words(self):
        segment = {"start": 0, "end": 2, "text": "text"}
        self.assertEqual(subtitles.remap_segments([], [(0, 2)]), [])
        self.assertEqual(subtitles.remap_segments([segment], []), [])
        for words in [[], None]:
            with self.subTest(words=words):
                source = dict(segment, words=words)
                self.assertEqual(subtitles.remap_segments([source], [(0, 2)]), [source])
        self.assertNotIn("words", subtitles.remap_segments([segment], [(0, 2)])[0])

    def test_iterators_and_overlapping_keeps(self):
        segments = iter([{"start": 1, "end": 9, "text": "kept"}])
        keeps = iter([(6, 10), (2, 7)])
        self.assertEqual(subtitles.remap_segments(segments, keeps),
                         [{"start": 0, "end": 7, "text": "kept"}])

    def test_invalid_segment_and_word_times(self):
        for start, end in [(3, 2), (0, float("inf")), (float("nan"), 3)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    subtitles.remap_segments(
                        [{"start": start, "end": end, "text": "bad"}], [(0, 10)]
                    )
                with self.assertRaises(ValueError):
                    subtitles.remap_segments([{
                        "start": 0, "end": 10, "text": "bad word",
                        "words": [{"start": start, "end": end, "word": "bad"}],
                    }], [(0, 10)])


class SrtTests(unittest.TestCase):
    def test_speaker_metadata_survives_remap_and_optional_prefix(self):
        segments = [{
            "start": 2, "end": 4, "text": " Labas", "speaker": "SPEAKER_00",
            "words": [{"start": 2, "end": 3, "word": "Labas", "speaker": "SPEAKER_01"}],
        }]
        original = deepcopy(segments)
        remapped = subtitles.remap_segments(segments, [(1, 5)])
        self.assertEqual(remapped[0]["speaker"], "SPEAKER_00")
        self.assertEqual(remapped[0]["words"][0]["speaker"], "SPEAKER_01")
        self.assertEqual(remapped[0]["start"], 1)
        self.assertNotIn("[SPEAKER_", subtitles.format_srt(remapped))
        with TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "speakers.srt"
            subtitles.write_srt(remapped, path, speaker_prefix=True)
            self.assertIn("[SPEAKER_00] Labas", path.read_text())
        self.assertEqual(segments, original)
        self.assertIn("plain\n\n", subtitles.format_srt(
            [{"start": 0, "end": 1, "text": "plain"}], speaker_prefix=True,
        ))

    def test_timestamp_rounding_and_normalized_carry(self):
        for seconds, expected in [
            (0, "00:00:00,000"), (1.2344, "00:00:01,234"),
            (1.2345, "00:00:01,235"), (0.0005, "00:00:00,001"),
            (59.9995, "00:01:00,000"), (3599.9995, "01:00:00,000"),
            (86399.9995, "24:00:00,000"), (360000, "100:00:00,000"),
            (Decimal("1.0015"), "00:00:01,002"),
        ]:
            with self.subTest(seconds=seconds):
                self.assertEqual(subtitles.format_timestamp(seconds), expected)

    def test_invalid_timestamps(self):
        for seconds in [-0.001, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(seconds=seconds):
                with self.assertRaises(ValueError):
                    subtitles.format_timestamp(seconds)
        with self.assertRaises(ValueError):
            subtitles.format_srt([{"start": 2, "end": 1, "text": "bad"}])

    def test_srt_numbering_multiline_unicode_and_empty(self):
        segments = [
            {"start": 0, "end": 1.2345, "text": "Héllo 世界\nSecond line"},
            {"start": 59.9995, "end": 61, "text": "Next", "id": 42},
        ]
        original = deepcopy(segments)
        self.assertEqual(subtitles.format_srt(iter(segments)),
                         "1\n00:00:00,000 --> 00:00:01,235\n"
                         "Héllo 世界\nSecond line\n\n"
                         "2\n00:01:00,000 --> 00:01:01,000\nNext\n\n")
        self.assertEqual(subtitles.format_srt([]), "")
        self.assertEqual(segments, original)

    def test_write_utf8_lf_and_overwrite_with_empty_srt(self):
        segments = [{"start": 0, "end": 1, "text": "Café 日本語"}]
        with TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "subtitles.srt"
            subtitles.write_srt(segments, path)
            self.assertEqual(path.read_bytes(),
                             "1\n00:00:00,000 --> 00:00:01,000\nCafé 日本語\n\n"
                             .encode("utf-8"))
            subtitles.write_srt([], str(path))
            self.assertEqual(path.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
