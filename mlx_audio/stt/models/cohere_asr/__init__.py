from .cohere_asr import Model
from .config import ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

DETECTION_HINTS = {
    "architectures": {
        "CohereAsrForConditionalGeneration",
        "CohereAsrModel",
    },
    "config_keys": {
        "max_audio_clip_s",
        "overlap_chunk_second",
        "supported_languages",
    },
}

__all__ = ["Model", "ModelConfig", "DETECTION_HINTS"]

ModelConfig.DOCS = ModelDocEntry(
    slug="cohere-transcribe",
    name="Cohere Transcribe",
    task="stt",
    description="Multilingual offline ASR with long-form chunking",
    repo="CohereLabs/cohere-transcribe-03-2026",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/cohere_asr/README.md",
    languages=("multilingual",),
    pipeline_tag="automatic-speech-recognition",
)
