from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fish_easy.drama3_runtime import build_request_payload


class Drama3RequestPayloadTests(unittest.TestCase):
    def build_payload(self, **overrides):
        options = {
            "text": "Hello there.",
            "reference_audio": None,
            "reference_text": None,
            "top_p": 0.9,
            "top_k": 30,
            "temperature": 1.0,
            "seed": 123,
            "max_tokens": 500,
            "context_length": 4096,
            "text_chunk_size": 200,
            "text_chunk_mode": "tag_aware",
        }
        options.update(overrides)
        return build_request_payload(**options)

    def test_payload_contains_sampling_and_chunk_controls(self):
        payload = self.build_payload()

        self.assertEqual(payload["input"], "Hello there.")
        self.assertEqual(payload["max_tokens"], 500)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["top_k"], 30)
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["seed"], 123)
        self.assertEqual(
            payload["options"],
            {"text_chunk_size": "200", "text_chunk_mode": "tag_aware"},
        )

    def test_payload_adds_reference_and_continuation_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "voice.wav"
            reference.touch()
            payload = self.build_payload(
                reference_audio=str(reference),
                reference_text="The exact reference transcript.",
                previous_takes='["Earlier line."]',
            )

        self.assertEqual(payload["voice_ref"], str(reference.resolve()))
        self.assertEqual(
            payload["options"]["reference_text"],
            "The exact reference transcript.",
        )
        self.assertEqual(payload["options"]["previous_takes"], '["Earlier line."]')

    def test_payload_accepts_multiple_references_and_history_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "history.json"
            history_file.write_text("[]", encoding="utf-8")
            payload = self.build_payload(
                multi_reference_cond='[{"text":"<|speaker:0|> Hi.","audio_path":"a.wav"}]',
                previous_takes_file=str(history_file),
            )

        self.assertIn("multi_reference_cond", payload["options"])
        self.assertEqual(
            payload["options"]["previous_takes_file"],
            str(history_file.resolve()),
        )

    def test_payload_rejects_context_overflow_and_malformed_history(self):
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            self.build_payload(max_tokens=600, context_length=500)
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            self.build_payload(previous_takes="{not-json}")

    def test_payload_requires_reference_transcript_and_exclusive_reference_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "voice.wav"
            reference.touch()
            with self.assertRaisesRegex(ValueError, "transcript"):
                self.build_payload(reference_audio=str(reference))
            with self.assertRaisesRegex(ValueError, "not both"):
                self.build_payload(
                    reference_audio=str(reference),
                    reference_text="Hello.",
                    multi_reference_cond="[]",
                )


if __name__ == "__main__":
    unittest.main()
