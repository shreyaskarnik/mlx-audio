from .bailingmm import Model, ModelConfig
from mlx_audio.model_catalog import ModelDocEntry

try:
    from .convert import convert_campplus_onnx_to_safetensors
except ModuleNotFoundError as exc:
    if exc.name != "onnx":
        raise

    _ONNX_IMPORT_ERROR = exc

    def convert_campplus_onnx_to_safetensors(*args, **kwargs):
        raise ModuleNotFoundError(
            "onnx is required for convert_campplus_onnx_to_safetensors function."
            "Please install onnx using `pip install onnx`."
        ) from _ONNX_IMPORT_ERROR


__all__ = ["Model", "ModelConfig", "convert_campplus_onnx_to_safetensors"]

ModelConfig.DOCS = ModelDocEntry(
    slug="ming-omni-bailingmm",
    name="Ming Omni TTS (BailingMM)",
    task="tts",
    description="Multimodal generation with voice cloning and style control",
    repo="mlx-community/Ming-omni-tts-16.8B-A3B-bf16",
    docs_path="https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/bailingmm/README.md",
    languages=("en", "zh"),
    tags=("multimodal", "voice-cloning"),
    pipeline_tag="text-to-speech",
    voice_cloning=True,
)
