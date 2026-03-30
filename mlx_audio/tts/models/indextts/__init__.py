from mlx_audio.tts.models.indextts.indextts import Model, ModelArgs
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig = ModelArgs

__all__ = ["Model", "ModelArgs", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="indextts",
    name="IndexTTS",
    task="tts",
    description="Zero-shot TTS with voice cloning and BigVGAN decoding",
    repo="mlx-community/IndexTTS",
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/indextts",
    languages=("multilingual",),
    tags=("voice-cloning",),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)
