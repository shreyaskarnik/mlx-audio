from .pocket_tts import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="pocket-tts",
    name="PocketTTS",
    task="tts",
    description="Compact streaming TTS with optional voice conditioning",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/pocket_tts",
    languages=("multilingual",),
    tags=("streaming", "voice-cloning"),
    pipeline_tag="text-to-speech",
    streaming=True,
    voice_cloning=True,
)
