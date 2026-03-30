from .dense import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig"]

ModelConfig.DOCS = ModelDocEntry(
    slug="ming-omni-dense",
    name="Ming Omni TTS (Dense)",
    task="tts",
    description="Lightweight Ming Omni variant for voice cloning and style control",
    repo="mlx-community/Ming-omni-tts-0.5B-bf16",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/dense/README.md",
    languages=("en", "zh"),
    tags=("multimodal", "voice-cloning"),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)
