from .config import ModelConfig
from .tada import Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="tada",
    name="TADA",
    task="tts",
    description="Text-Acoustic Dual Alignment TTS with voice cloning",
    repo="HumeAI/mlx-tada-1b",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/tada/README.md",
    languages=("en", "multilingual"),
    tags=("voice-cloning",),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)
