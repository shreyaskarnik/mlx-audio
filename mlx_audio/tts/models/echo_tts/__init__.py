from .echo_tts import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="echo-tts",
    name="Echo TTS",
    task="tts",
    description="Diffusion-based TTS with fast voice cloning",
    repo="mlx-community/echo-tts-base",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/echo_tts/README.md",
    languages=("en",),
    tags=("diffusion", "voice-cloning"),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)

__all__ = ["Model", "ModelConfig"]
