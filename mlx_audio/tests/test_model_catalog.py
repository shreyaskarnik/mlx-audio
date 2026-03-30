import unittest

from mlx_audio.model_catalog import (
    CATALOG_EXCLUDED_PACKAGES,
    collect_model_doc_entries,
    get_model_doc_entries,
    get_model_doc_entry,
    iter_model_packages,
)


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

    def test_get_model_doc_entries_from_package_init(self):
        entries = get_model_doc_entries("mlx_audio.tts.models.chatterbox")

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].slug, "chatterbox")
        self.assertTrue(entries[0].voice_cloning)

    def test_get_model_doc_entries_supports_multiple_entries(self):
        entries = get_model_doc_entries("mlx_audio.stt.models.qwen3_asr")

        self.assertEqual(
            [entry.slug for entry in entries],
            ["qwen3-asr", "qwen3-forced-aligner"],
        )

    def test_iter_model_packages_skips_internal_catalog_exclusions(self):
        packages = set(iter_model_packages())

        self.assertTrue(CATALOG_EXCLUDED_PACKAGES.isdisjoint(packages))

    def test_all_catalog_packages_have_metadata(self):
        missing = [
            package
            for package in iter_model_packages()
            if not get_model_doc_entries(package)
        ]

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
