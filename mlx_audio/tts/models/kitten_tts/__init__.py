from .kitten_tts import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="kitten-tts",
    name="KittenTTS",
    task="tts",
    description="Small English TTS with preset expressive voices",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/kitten_tts",
    languages=("en",),
    tags=("preset-voices",),
    pipeline_tag="text-to-speech",
    voice_cloning=False,
)
