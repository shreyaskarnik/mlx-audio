from .config import ModelConfig
from .qwen2_audio import Model
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="qwen2-audio",
    name="Qwen2-Audio",
    task="stt",
    description="Audio-language model for transcription, translation, and audio understanding",
    repo="mlx-community/Qwen2-Audio-7B-Instruct-4bit",
    docs_path="models/stt/qwen2-audio/",
    languages=("multilingual",),
    tags=("translation", "audio-understanding"),
    pipeline_tag="automatic-speech-recognition",
)

__all__ = ["Model", "ModelConfig"]
