from .config import (
    AcousticTokenizerConfig,
    DiffusionHeadConfig,
    ModelConfig,
    Qwen2DecoderConfig,
)
from .vibevoice import Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = [
    "Model",
    "ModelConfig",
    "AcousticTokenizerConfig",
    "DiffusionHeadConfig",
    "Qwen2DecoderConfig",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="vibevoice",
    name="VibeVoice",
    task="tts",
    description="Streaming multilingual TTS with cached voice conditioning",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/vibevoice",
    languages=("multilingual",),
    tags=("streaming", "voice-cloning"),
    pipeline_tag="text-to-speech",
    streaming=True,
    voice_cloning=True,
)
