from .config import AudioConfig, ModelConfig, TextConfig
from .voxtral import Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = [
    "AudioConfig",
    "TextConfig",
    "ModelConfig",
    "Model",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="voxtral",
    name="Voxtral",
    task="stt",
    description="Multilingual speech model from Mistral",
    repo="mlx-community/Voxtral-Mini-3B-2507-bf16",
    docs_path="https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16",
    languages=("multilingual",),
    pipeline_tag="automatic-speech-recognition",
)
