from .config import ModelArgs
from .voxcpm import Model
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig = ModelArgs

__all__ = ["Model", "ModelConfig", "ModelArgs"]

ModelConfig.DOCS = ModelDocEntry(
    slug="voxcpm",
    name="VoxCPM",
    task="tts",
    description="Multilingual TTS with voice cloning and 44.1kHz audio output",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/voxcpm",
    languages=("multilingual",),
    tags=("voice-cloning", "high-sample-rate"),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)
