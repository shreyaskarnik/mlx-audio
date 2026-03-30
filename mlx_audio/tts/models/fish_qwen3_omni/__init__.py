from .config import ModelConfig
from .fish_speech import Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="fish-speech",
    name="Fish Speech",
    task="tts",
    description="Voice cloning and multi-speaker TTS with inline control tags",
    repo="mlx-community/fish-audio-s2-pro",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/fish_qwen3_omni/README.md",
    languages=("multilingual",),
    tags=("voice-cloning", "multi-speaker"),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)
