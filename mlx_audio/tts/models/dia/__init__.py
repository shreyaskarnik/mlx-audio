from .dia import Model
from .config import ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="dia",
    name="Dia",
    task="tts",
    description="Dialogue-focused TTS with speaker tags",
    repo="mlx-community/Dia-1.6B-fp16",
    docs_path="models/tts/dia/",
    languages=("en",),
    tags=("dialogue", "multi-speaker"),
    pipeline_tag="text-to-speech",
)

__all__ = ["Model", "ModelConfig"]
