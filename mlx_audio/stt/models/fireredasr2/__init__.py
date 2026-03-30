from .config import ModelConfig
from .fireredasr2 import FireRedASR2
from .fireredasr2 import FireRedASR2 as Model
from mlx_audio.model_catalog import ModelDocEntry

ModelConfig.DOCS = ModelDocEntry(
    slug="fireredasr2",
    name="FireRedASR2",
    task="stt",
    description="Bilingual Chinese and English ASR",
    repo="mlx-community/FireRedASR2-AED-mlx",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/fireredasr2/README.md",
    languages=("zh", "en"),
    pipeline_tag="automatic-speech-recognition",
)

__all__ = ["Model", "ModelConfig", "FireRedASR2"]
