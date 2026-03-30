from .moshi import MoshiConfig, MoshiSTSModel
from mlx_audio.model_catalog import ModelDocEntry

Model = MoshiSTSModel
ModelConfig = MoshiConfig

ModelConfig.DOCS = ModelDocEntry(
    slug="moshi",
    name="Moshi",
    task="sts",
    description="Full-duplex voice conversation model",
    repo="kyutai/moshiko-mlx-q4",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/moshi/README.md",
    tags=("conversation",),
    streaming=True,
)

__all__ = ["MoshiSTSModel", "MoshiConfig", "Model", "ModelConfig"]
