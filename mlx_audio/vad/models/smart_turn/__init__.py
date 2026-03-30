from .config import EncoderConfig, ModelConfig, ProcessorConfig
from .smart_turn import EndpointOutput, Model
from mlx_audio.model_catalog import ModelDocEntry

DETECTION_HINTS = {
    "architectures": ["smart_turn"],
    "config_keys": ["max_audio_seconds", "encoder_config", "processor_config"],
}

__all__ = [
    "EncoderConfig",
    "ProcessorConfig",
    "ModelConfig",
    "EndpointOutput",
    "Model",
    "DETECTION_HINTS",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="smart-turn",
    name="Smart Turn",
    task="vad",
    description="Endpoint detection for conversational turn-taking",
    repo="mlx-community/smart-turn-v3",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/smart_turn/README.md",
    tags=("endpoint-detection",),
)
