from .config import ModelConfig
from .kugelaudio import Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="kugelaudio",
    name="KugelAudio",
    task="tts",
    description="7B multilingual TTS for European languages with diffusion decoding",
    repo="kugelaudio/kugelaudio-0-open",
    docs_path="models/tts/kugelaudio/",
    languages=(
        "en",
        "de",
        "fr",
        "es",
        "it",
        "pt",
        "nl",
        "pl",
        "ru",
        "uk",
        "cs",
        "ro",
        "hu",
        "sv",
        "da",
        "fi",
        "no",
        "el",
        "bg",
        "sk",
        "hr",
        "sr",
        "tr",
    ),
    license="mit",
    pipeline_tag="text-to-speech",
)
