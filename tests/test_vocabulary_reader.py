"""CPU-only checks of vocabulary ordering, exact gaps and resumable failures."""

from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import soundfile as sf

from tools.vocabulary_audio import assemble_audio, prepare_audio, read_mono, trim_edges, write_wav
from tools.vocabulary_reader import VocabularyReader, audio_options, parse_vocabulary


class FakeModel:
    def __init__(self):
        self.calls = []
        self.fail = set()

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["text"] in self.fail:
            raise RuntimeError("synthetic failure")
        write_wav(kwargs["output_path"], np.full(100, 0.2, dtype=np.float32), 1000)


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.wav"
        write_wav(self.reference, np.full(200, 0.1, dtype=np.float32), 1000)
        self.model = FakeModel()
        self.lock = threading.Lock()
        self.reader = VocabularyReader(self.model, self.root / "jobs", self.lock)

    def test_parse_keeps_duplicates_and_warns_about_teaching_markup(self):
        items, warnings = parse_vocabulary(" ignore \n\ngap\nignore\nonly child ")
        self.assertEqual(items, ["ignore", "gap", "ignore", "only child"])
        self.assertFalse(warnings)
        self.assertEqual(len(parse_vocabulary("[停顿]\n1. ignore\nbe into sth")[1]), 3)

    def test_exact_samples_gaps_and_no_trailing_gap(self):
        a, b, output = (self.root / name for name in ("a.wav", "b.wav", "mix.wav"))
        write_wav(a, np.full(100, 0.2), 1000)
        write_wav(b, np.full(200, 0.4), 1000)
        stats = assemble_audio([a, b], output, 2, 0.0804, 0.1504)
        audio, rate = read_mono(output)
        self.assertEqual(rate, 1000)
        self.assertEqual(len(audio), 2 * (100 + 200) + 2 * 80 + 150)
        self.assertEqual(stats["samples"], len(audio))
        self.assertTrue(np.all(audio[100:180] == 0))
        self.assertTrue(np.all(audio[280:430] == 0))
        self.assertGreater(audio[-1], 0)

    def test_prepare_downmixes_resamples_and_preserves_source(self):
        source, destination = self.root / "stereo.wav", self.root / "mono.wav"
        sf.write(source, np.full((200, 2), 0.2), 2000)
        before = source.read_bytes()
        rate, count = prepare_audio(source, destination, sample_rate=1000)
        self.assertEqual((rate, count), (1000, 100))
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(sf.info(destination).channels, 1)

    def test_trim_keeps_quiet_edges_and_margin(self):
        audio = np.r_[np.zeros(300), np.full(20, 0.0005), np.full(100, 0.2),
                      np.full(20, 0.0005), np.zeros(300)].astype(np.float32)
        trimmed = trim_edges(audio, 1000)
        self.assertEqual(len(trimmed), 300)
        self.assertTrue(np.allclose(trimmed[80:100], 0.0005))
        self.assertTrue(np.allclose(trimmed[-100:-80], 0.0005))
        np.testing.assert_array_equal(trim_edges(np.zeros(100), 1000), np.zeros(100))

    def test_reassemble_reuses_inference_and_selected_regeneration_is_isolated(self):
        task_id = self.reader.create("ignore\ngap\nignore", self.reference)
        task = list(self.reader.run(task_id))[-1]
        self.assertEqual(task["status"], "complete")
        self.assertEqual([call["text"] for call in self.model.calls], ["ignore", "gap", "ignore"])
        initial_raw = [item["raw"] for item in task["items"]]
        original_export = task["combined"]
        task = self.reader.reassemble(task_id, repeats=3, repeat_gap=0.1, item_gap=0.2, trim=False)
        self.assertEqual(len(self.model.calls), 3)
        self.assertEqual(task["exports"][-1]["samples"], 3 * 3 * 100 + 3 * 2 * 100 + 2 * 200)
        self.assertNotEqual(task["combined"], original_export)
        self.assertTrue((self.reader.directory(task_id) / original_export).exists())
        task = list(self.reader.run(task_id, [1]))[-1]
        self.assertEqual(len(self.model.calls), 4)
        self.assertEqual([item["revision"] for item in task["items"]], [1, 2, 1])
        self.assertEqual(task["items"][0]["raw"], initial_raw[0])
        self.assertTrue((self.reader.directory(task_id) / initial_raw[1]).exists())

    def test_failed_item_blocks_export_then_retry_keeps_successes(self):
        self.model.fail.add("gap")
        task_id = self.reader.create("ignore\ngap\ncode", self.reference)
        task = list(self.reader.run(task_id))[-1]
        self.assertEqual(task["status"], "incomplete")
        self.assertIsNone(task["combined"])
        self.assertIn("synthetic failure", task["items"][1]["error"])
        with self.assertRaises(ValueError):
            self.reader.reassemble(task_id)
        self.model.fail.clear()
        task = list(self.reader.run(task_id))[-1]
        self.assertEqual(task["status"], "complete")
        self.assertEqual([call["text"] for call in self.model.calls], ["ignore", "gap", "code", "gap"])
        self.assertEqual(len(task["items"][1]["history"]), 2)

    def test_invalid_inputs_and_task_ids(self):
        for text in ("", "[停顿]", "be into sth"):
            with self.assertRaises(ValueError):
                self.reader.create(text, self.reference)
        with self.assertRaises(ValueError):
            self.reader.load("../../anything")
        for repeats in (0, 1.5, float("inf")):
            with self.assertRaises(ValueError):
                audio_options(repeats=repeats)

    def test_cancelled_generator_marks_task_for_retry(self):
        task_id = self.reader.create("ignore", self.reference)
        stream = self.reader.run(task_id)
        next(stream)
        stream.close()
        self.assertEqual(self.reader.load(task_id)["status"], "incomplete")
        self.assertEqual(list(self.reader.run(task_id))[-1]["status"], "complete")

    def test_failed_regeneration_withdraws_current_export(self):
        task_id = self.reader.create("ignore\ngap", self.reference)
        task = list(self.reader.run(task_id))[-1]
        old_export = task["combined"]
        self.model.fail.add("gap")
        task = list(self.reader.run(task_id, [1]))[-1]
        self.assertIsNone(task["combined"])
        self.assertEqual(task["items"][1]["status"], "failed")
        self.assertTrue((self.reader.directory(task_id) / old_export).exists())

    def test_assembly_failure_records_error_and_withdraws_export(self):
        task_id = self.reader.create("ignore", self.reference)
        task = list(self.reader.run(task_id))[-1]
        (self.reader.directory(task_id) / task["items"][0]["raw"]).unlink()
        with self.assertRaises(Exception):
            self.reader.reassemble(task_id)
        task = self.reader.load(task_id)
        self.assertIsNone(task["combined"])
        self.assertEqual(task["status"], "assembly_failed")
        self.assertTrue(task["assembly_error"])


if __name__ == "__main__":
    unittest.main()
