from .parakeet import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="parakeet",
    name="Parakeet",
    task="stt",
    description="Fast multilingual ASR with streaming support",
    repo="mlx-community/parakeet-tdt-0.6b-v3",
    docs_path="models/stt/parakeet/",
    languages=("multilingual",),
    tags=("word-timestamps", "streaming"),
    pipeline_tag="automatic-speech-recognition",
    streaming=True,
    timestamps=True,
)

__all__ = ["Model", "ModelConfig"]
