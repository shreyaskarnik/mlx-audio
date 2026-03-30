from .config import AudioEncoderConfig, ModelConfig, TextConfig
from .qwen3_asr import Model, Qwen3ASRModel, StreamingResult
from .qwen3_forced_aligner import (
    ForceAlignProcessor,
    ForcedAlignerConfig,
    ForcedAlignerModel,
    ForcedAlignItem,
    ForcedAlignResult,
)
from mlx_audio.model_catalog import ModelDocEntry

__all__ = [
    "AudioEncoderConfig",
    "TextConfig",
    "ModelConfig",
    "Model",
    "Qwen3ASRModel",
    "StreamingResult",
    "ForcedAlignerConfig",
    "ForcedAlignerModel",
    "ForcedAlignItem",
    "ForcedAlignResult",
    "ForceAlignProcessor",
]

ModelConfig.DOCS = (
    ModelDocEntry(
        slug="qwen3-asr",
        name="Qwen3-ASR",
        task="stt",
        description="Multilingual ASR with streaming support",
        repo="mlx-community/Qwen3-ASR-1.7B-8bit",
        docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md",
        languages=("zh", "en", "ja", "ko", "multilingual"),
        tags=("streaming",),
        pipeline_tag="automatic-speech-recognition",
        streaming=True,
        timestamps=True,
    ),
    ModelDocEntry(
        slug="qwen3-forced-aligner",
        name="Qwen3-ForcedAligner",
        task="stt",
        description="Word-level forced alignment for speech transcripts",
        repo="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
        docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md",
        languages=("zh", "en", "ja", "ko", "multilingual"),
        tags=("alignment", "word-timestamps"),
        pipeline_tag="automatic-speech-recognition",
        timestamps=True,
    ),
)
