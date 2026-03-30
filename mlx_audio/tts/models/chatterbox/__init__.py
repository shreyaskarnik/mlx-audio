from .chatterbox import Model
from .config import ModelConfig
from .scripts.convert import convert_from_source
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig", "convert_from_source"]

ModelConfig.DOCS = ModelDocEntry(
    slug="chatterbox",
    name="Chatterbox",
    task="tts",
    description="Expressive multilingual TTS with voice cloning",
    repo="mlx-community/chatterbox-fp16",
    docs_path="models/tts/chatterbox/",
    languages=("en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"),
    tags=("expressive", "voice-cloning"),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)
