from .soprano import DecoderConfig, Model, ModelConfig
from .text import clean_text
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="soprano",
    name="Soprano",
    task="tts",
    description="High-quality English TTS",
    repo="mlx-community/Soprano-1.1-80M-bf16",
    docs_path="https://huggingface.co/mlx-community/Soprano-1.1-80M-bf16",
    languages=("en",),
    pipeline_tag="text-to-speech",
)

__all__ = ["DecoderConfig", "Model", "ModelConfig", "clean_text"]
