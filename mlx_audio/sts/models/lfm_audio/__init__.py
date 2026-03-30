# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from .config import (
    ConformerEncoderConfig,
    DepthformerConfig,
    DetokenizerConfig,
    LFM2AudioConfig,
    LFM2Config,
    PreprocessorConfig,
)
from .detokenizer import LFM2AudioDetokenizer
from .model import GenerationConfig, LFM2AudioModel, LFMModality
from .processor import AudioPreprocessor, ChatState, LFM2AudioProcessor
from mlx_audio.model_catalog import ModelDocEntry

Model = LFM2AudioModel  # Alias for LFM2AudioModel
ModelConfig = LFM2AudioConfig  # Alias for LFM2AudioConfig

__all__ = [
    # Config
    "LFM2AudioConfig",
    "LFM2Config",
    "ConformerEncoderConfig",
    "DepthformerConfig",
    "PreprocessorConfig",
    "DetokenizerConfig",
    # Model
    "LFM2AudioModel",
    "LFMModality",
    "GenerationConfig",
    # Processor
    "LFM2AudioProcessor",
    "AudioPreprocessor",
    "LFM2AudioDetokenizer",
    "ChatState",
    "Model",
    "ModelConfig",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="lfm2-5-audio",
    name="Liquid2.5-Audio",
    task="sts",
    description="Multimodal speech-to-speech, TTS, and ASR in one model",
    repo="mlx-community/LFM2.5-Audio-1.5B-4bit",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/lfm_audio/README.md",
    tags=("multimodal",),
)
