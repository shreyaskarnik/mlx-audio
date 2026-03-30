from .config import FCEncoderConfig, ModelConfig, ModulesConfig, TFEncoderConfig
from .sortformer import Model, StreamingState
from mlx_audio.model_catalog import ModelDocEntry

DETECTION_HINTS = {
    "architectures": ["SortformerOffline"],
    "config_keys": ["fc_encoder_config", "tf_encoder_config", "sortformer_modules"],
}

__all__ = [
    "FCEncoderConfig",
    "TFEncoderConfig",
    "ModulesConfig",
    "ModelConfig",
    "Model",
    "DETECTION_HINTS",
]

ModelConfig.DOCS = (
    ModelDocEntry(
        slug="sortformer-v1",
        name="Sortformer v1",
        task="vad",
        description="End-to-end speaker diarization for up to four speakers",
        repo="mlx-community/diar_sortformer_4spk-v1-fp32",
        docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/sortformer/README.md",
        tags=("speaker-diarization",),
        diarization=True,
    ),
    ModelDocEntry(
        slug="sortformer-v2-1",
        name="Sortformer v2.1",
        task="vad",
        description="Streaming speaker diarization with AOSC compression",
        repo="mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32",
        docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/sortformer/README.md",
        tags=("speaker-diarization", "streaming"),
        streaming=True,
        diarization=True,
    ),
)
