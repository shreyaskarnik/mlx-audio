from .bark import Model, ModelConfig
from .pipeline import Pipeline
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "Pipeline", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="bark",
    name="Bark",
    task="tts",
    description="Promptable multilingual TTS with preset voice prompts",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/bark",
    languages=("multilingual",),
    tags=("prompt-voices",),
    pipeline_tag="text-to-speech",
    voice_cloning=False,
)
