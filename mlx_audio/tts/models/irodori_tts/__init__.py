from .irodori_tts import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="irodori-tts",
    name="Irodori TTS",
    task="tts",
    description="Japanese TTS with DiT and DACVAE decoding",
    repo="mlx-community/Irodori-TTS-500M-fp16",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/irodori_tts/README.md",
    languages=("ja",),
    tags=("voice-cloning",),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)

__all__ = ["Model", "ModelConfig"]
