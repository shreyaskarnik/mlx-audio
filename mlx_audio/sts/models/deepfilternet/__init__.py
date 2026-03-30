"""DeepFilterNet speech enhancement model for MLX."""

from .config import DeepFilterNet2Config, DeepFilterNet3Config, DeepFilterNetConfig
from .model import DeepFilterNetModel
from .streaming import DeepFilterNetStreamer, DeepFilterNetStreamingConfig
from mlx_audio.model_catalog import ModelDocEntry

Model = DeepFilterNetModel
ModelConfig = DeepFilterNetConfig

__all__ = [
    "DeepFilterNetModel",
    "DeepFilterNetConfig",
    "DeepFilterNet2Config",
    "DeepFilterNet3Config",
    "DeepFilterNetStreamer",
    "DeepFilterNetStreamingConfig",
    "Model",
    "ModelConfig",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="deepfilternet",
    name="DeepFilterNet",
    task="sts",
    description="Speech enhancement and noise suppression",
    repo="mlx-community/DeepFilterNet-mlx",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/deepfilternet/README.md",
    tags=("speech-enhancement", "streaming"),
    streaming=True,
)
