from .config import EncoderConfig, ModelConfig, ProjectorConfig, TextConfig
from .granite_speech import Model
from mlx_audio.model_catalog import ModelDocEntry

DETECTION_HINTS = {
    "config_keys": {"encoder_config", "projector_config", "audio_token_index"},
    "architectures": {"GraniteSpeechForConditionalGeneration"},
}

__all__ = [
    "EncoderConfig",
    "ProjectorConfig",
    "TextConfig",
    "ModelConfig",
    "Model",
    "DETECTION_HINTS",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="granite-speech",
    name="Granite Speech",
    task="stt",
    description="ASR and speech translation from IBM Granite",
    repo="ibm-granite/granite-4.0-1b-speech",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/granite_speech/README.md",
    languages=("en", "fr", "de", "es", "pt", "ja"),
    pipeline_tag="automatic-speech-recognition",
    streaming=True,
)
