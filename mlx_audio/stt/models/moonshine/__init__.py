from .moonshine import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="moonshine",
    name="Moonshine",
    task="stt",
    description="Lightweight English ASR",
    repo="UsefulSensors/moonshine-base",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/moonshine/README.md",
    languages=("en",),
    pipeline_tag="automatic-speech-recognition",
)

__all__ = ["Model", "ModelConfig"]
