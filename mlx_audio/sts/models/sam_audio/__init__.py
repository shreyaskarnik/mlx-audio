# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from .config import DACVAEConfig, SAMAudioConfig, T5EncoderConfig, TransformerConfig
from .model import SAMAudio, SeparationResult
from .processor import Batch, SAMAudioProcessor, save_audio
from mlx_audio.model_catalog import ModelDocEntry

Model = SAMAudio
ModelConfig = SAMAudioConfig

__all__ = [
    "SAMAudio",
    "SAMAudioProcessor",
    "SeparationResult",
    "Batch",
    "save_audio",
    "SAMAudioConfig",
    "DACVAEConfig",
    "T5EncoderConfig",
    "TransformerConfig",
    "Model",
    "ModelConfig",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="sam-audio",
    name="SAM-Audio",
    task="sts",
    description="Text-guided audio source separation",
    repo="mlx-community/sam-audio-large",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/sam_audio/README.md",
    tags=("source-separation",),
)
