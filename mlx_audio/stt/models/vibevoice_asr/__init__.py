# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from .config import (
    AcousticTokenizerConfig,
    ModelConfig,
    Qwen2Config,
    SemanticTokenizerConfig,
)
from .vibevoice_asr import Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = [
    "Model",
    "ModelConfig",
    "AcousticTokenizerConfig",
    "SemanticTokenizerConfig",
    "Qwen2Config",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="vibevoice-asr",
    name="VibeVoice-ASR",
    task="stt",
    description="Multilingual ASR with diarization support",
    repo="mlx-community/VibeVoice-ASR-bf16",
    docs_path="https://huggingface.co/mlx-community/VibeVoice-ASR-bf16",
    languages=("multilingual",),
    tags=("diarization", "streaming"),
    pipeline_tag="automatic-speech-recognition",
    streaming=True,
    diarization=True,
)
