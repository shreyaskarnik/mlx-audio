from .sesame import Model
from .sesame import SesameModelArgs as ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="csm",
    name="CSM",
    task="tts",
    description="Conversational speech model with voice cloning",
    repo="mlx-community/csm-1b",
    docs_path="models/tts/csm/",
    languages=("en",),
    tags=("voice-cloning", "streaming", "conversational"),
    pipeline_tag="text-to-speech",
    streaming=True,
    voice_cloning=True,
)

__all__ = ["Model", "ModelConfig"]
