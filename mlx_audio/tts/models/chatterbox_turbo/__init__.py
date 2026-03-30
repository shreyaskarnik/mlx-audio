# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from .chatterbox_turbo import ChatterboxTurboTTS, Conditionals, punc_norm
from .models.s3gen import S3GEN_SIL, S3GEN_SR, S3Gen
from .models.t3 import T3, T3Cond, T3Config
from .models.voice_encoder import VoiceEncoder
from mlx_audio.model_catalog import ModelDocEntry

# Alias for load_model compatibility
Model = ChatterboxTurboTTS
ModelConfig = T3Config

__all__ = [
    "ChatterboxTurboTTS",
    "Model",
    "ModelConfig",
    "Conditionals",
    "punc_norm",
    "T3",
    "T3Config",
    "T3Cond",
    "S3Gen",
    "S3GEN_SR",
    "S3GEN_SIL",
    "VoiceEncoder",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="chatterbox-turbo",
    name="Chatterbox Turbo",
    task="tts",
    description="Low-latency expressive TTS with voice cloning and streaming",
    repo="ResembleAI/chatterbox-turbo",
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/chatterbox_turbo",
    languages=("en",),
    tags=("voice-cloning", "streaming", "low-latency"),
    pipeline_tag="text-to-speech",
    streaming=True,
    voice_cloning=True,
)
