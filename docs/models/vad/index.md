<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->

# Voice Activity Detection Models

MLX Audio includes endpoint detection and diarization models for conversational turn-taking and speaker-aware processing.

## Model Catalog

| Model | Description | Streaming | Diarization | Repo |
|-------|-------------|:---------:|:-----------:|------|
| [**Smart Turn**](generated/smart-turn.md) | Endpoint detection for conversational turn-taking | -- | -- | [mlx-community/smart-turn-v3](https://huggingface.co/mlx-community/smart-turn-v3) |
| [**Sortformer v1**](generated/sortformer-v1.md) | End-to-end speaker diarization for up to four speakers | -- | Yes | [mlx-community/diar_sortformer_4spk-v1-fp32](https://huggingface.co/mlx-community/diar_sortformer_4spk-v1-fp32) |
| [**Sortformer v2.1**](generated/sortformer-v2-1.md) | Streaming speaker diarization with AOSC compression | Yes | Yes | [mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32) |

!!! tip "Choosing a model"
    - **Turn-taking / endpointing:** Smart Turn is the dedicated endpoint detector.
    - **Diarization:** Sortformer variants handle speaker attribution, with v2.1 optimized for streaming.
