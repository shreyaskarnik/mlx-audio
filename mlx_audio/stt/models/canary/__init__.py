from .canary import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="canary",
    name="Canary",
    task="stt",
    description="Multilingual ASR with speech translation",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/canary/README.md",
    languages=("multilingual",),
    pipeline_tag="automatic-speech-recognition",
)
