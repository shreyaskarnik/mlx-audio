from .spark import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="spark-tts",
    name="Spark",
    task="tts",
    description="SparkTTS model for English and Chinese speech generation",
    repo="mlx-community/Spark-TTS-0.5B-bf16",
    docs_path="https://huggingface.co/mlx-community/Spark-TTS-0.5B-bf16",
    languages=("en", "zh"),
    pipeline_tag="text-to-speech",
)

__all__ = ["Model", "ModelConfig"]
