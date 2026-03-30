import unittest

from mlx_audio.model_catalog import collect_model_doc_entries, get_model_doc_entry


class TestModelCatalog(unittest.TestCase):
    def test_get_model_doc_entry_from_model_config_classvar(self):
        entry = get_model_doc_entry("mlx_audio.tts.models.voxtral_tts")

        self.assertIsNotNone(entry)
        self.assertEqual(entry.slug, "voxtral-tts")
        self.assertEqual(entry.task, "tts")
        self.assertTrue(entry.streaming)
        self.assertEqual(entry.languages, ("en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"))

    def test_collect_model_doc_entries_from_explicit_packages(self):
        entries = collect_model_doc_entries(
            packages=[
                "mlx_audio.tts.models.kokoro",
                "mlx_audio.tts.models.voxtral_tts",
                "mlx_audio.stt.models.whisper",
            ],
            ignore_import_errors=False,
        )

        self.assertEqual(
            [entry.slug for entry in entries], ["whisper", "kokoro", "voxtral-tts"]
        )
        self.assertTrue(
            any(entry.timestamps for entry in entries if entry.slug == "whisper")
        )
        self.assertEqual(
            next(entry.pipeline_tag for entry in entries if entry.slug == "whisper"),
            "automatic-speech-recognition",
        )


if __name__ == "__main__":
    unittest.main()
