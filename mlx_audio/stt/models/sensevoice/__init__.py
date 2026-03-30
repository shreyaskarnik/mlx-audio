from .config import ModelConfig
from .sensevoice import SenseVoiceSmall
from .sensevoice import SenseVoiceSmall as Model
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="sensevoice",
    name="SenseVoice",
    task="stt",
    description="Multilingual speech recognition with emotion and event detection",
    repo="mlx-community/SenseVoiceSmall",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/README.md",
    languages=("multilingual",),
    tags=("emotion-detection", "event-detection"),
    pipeline_tag="automatic-speech-recognition",
)

__all__ = ["Model", "ModelConfig", "SenseVoiceSmall"]
