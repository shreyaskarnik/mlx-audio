from .config import AudioEncodingConfig, DecoderConfig, EncoderConfig, ModelConfig
from .voxtral_realtime import Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = [
    "AudioEncodingConfig",
    "DecoderConfig",
    "EncoderConfig",
    "ModelConfig",
    "Model",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="voxtral-realtime",
    name="Voxtral Realtime",
    task="stt",
    description="Streaming multilingual speech-to-text",
    repo="mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
    docs_path="models/stt/voxtral-realtime/",
    languages=("multilingual",),
    tags=("streaming",),
    pipeline_tag="automatic-speech-recognition",
    streaming=True,
)
