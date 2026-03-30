"""
MossFormer2 SE speech enhancement model for MLX.

This module provides a speech enhancement model based on MossFormer2 architecture,
optimized for 48kHz audio on Apple Silicon.
"""

# Reuse audio utilities from sam_audio
from ..sam_audio.processor import load_audio, save_audio
from .config import MossFormer2SEConfig
from .model import MossFormer2SEModel
from .mossformer2_se_wrapper import MossFormer2SE
from mlx_audio.model_catalog import ModelDocEntry

Model = MossFormer2SEModel
ModelConfig = MossFormer2SEConfig

__all__ = [
    "MossFormer2SEConfig",
    "MossFormer2SE",
    "MossFormer2SEModel",
    "load_audio",
    "save_audio",
    "Model",
    "ModelConfig",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="mossformer2-se",
    name="MossFormer2 SE",
    task="sts",
    description="Speech enhancement and noise removal",
    repo="starkdmi/MossFormer2_SE_48K_MLX",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/mossformer2_se/README.md",
    tags=("speech-enhancement",),
)
