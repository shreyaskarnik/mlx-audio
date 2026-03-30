from .config import ModelConfig
from .lasr import LasrForCTC
from .lasr import LasrForCTC as Model
from mlx_audio.model_catalog import ModelDocEntry

__all__ = ["Model", "ModelConfig", "LasrForCTC"]

ModelConfig.DOCS = ModelDocEntry(
    slug="lasr-ctc",
    name="LASR-CTC",
    task="stt",
    description="CTC-based speech recognition with a LASR encoder",
    repo=None,
    docs_path="https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/stt/models/lasr_ctc",
    pipeline_tag="automatic-speech-recognition",
)
