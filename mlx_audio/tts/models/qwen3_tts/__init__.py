from .qwen3_tts import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="qwen3-tts",
    name="Qwen3-TTS",
    task="tts",
    description="Multilingual TTS with voice cloning and voice design",
    repo="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    docs_path="models/tts/qwen3-tts/",
    languages=("zh", "en", "ja", "ko", "multilingual"),
    tags=("voice-cloning", "voice-design", "streaming"),
    pipeline_tag="text-to-speech",
    streaming=True,
    voice_cloning=True,
)
