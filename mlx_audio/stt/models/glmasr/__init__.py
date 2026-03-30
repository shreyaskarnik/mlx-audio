from .config import LlamaConfig, ModelConfig, WhisperConfig
from .glmasr import Model, StreamingResult, STTOutput
from mlx_audio.model_catalog import ModelDocEntry

__all__ = [
    "Model",
    "ModelConfig",
    "WhisperConfig",
    "LlamaConfig",
    "STTOutput",
    "StreamingResult",
]

ModelConfig.DOCS = ModelDocEntry(
    slug="glm-asr",
    name="GLM-ASR",
    task="stt",
    description="Speech recognition with a Whisper encoder and GLM decoder",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/stt/models/glmasr",
    languages=("zh", "en"),
    pipeline_tag="automatic-speech-recognition",
)
