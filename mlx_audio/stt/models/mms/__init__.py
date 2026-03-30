from .mms import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="mms",
    name="MMS",
    task="stt",
    description="Massively multilingual speech recognition",
    repo="facebook/mms-1b-all",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/README.md",
    languages=("multilingual",),
    pipeline_tag="automatic-speech-recognition",
)

__all__ = ["Model", "ModelConfig"]
