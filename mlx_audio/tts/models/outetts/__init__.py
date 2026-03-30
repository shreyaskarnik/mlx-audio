from .outetts import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="outetts",
    name="OuteTTS",
    task="tts",
    description="Efficient text-to-speech for Apple Silicon",
    repo="mlx-community/OuteTTS-1.0-0.6B-fp16",
    docs_path="https://huggingface.co/mlx-community/OuteTTS-1.0-0.6B-fp16",
    languages=("en",),
    pipeline_tag="text-to-speech",
)

__all__ = ["Model", "ModelConfig"]
