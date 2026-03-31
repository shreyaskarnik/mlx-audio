<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->

# Speech-to-Speech Models

MLX Audio includes speech enhancement, source separation, and multimodal speech interaction models under the STS umbrella.

## Model Catalog

| Model | Description | Repo |
|-------|-------------|------|
| [**DeepFilterNet**](generated/deepfilternet.md) | Speech enhancement and noise suppression | [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx) |
| [**Liquid2.5-Audio**](generated/lfm2-5-audio.md) | Multimodal speech-to-speech, TTS, and ASR in one model | [mlx-community/LFM2.5-Audio-1.5B-4bit](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-4bit) |
| [**Moshi**](generated/moshi.md) | Full-duplex voice conversation model | [kyutai/moshiko-mlx-q4](https://huggingface.co/kyutai/moshiko-mlx-q4) |
| [**MossFormer2 SE**](generated/mossformer2-se.md) | Speech enhancement and noise removal | [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX) |
| [**SAM-Audio**](generated/sam-audio.md) | Text-guided audio source separation | [mlx-community/sam-audio-large](https://huggingface.co/mlx-community/sam-audio-large) |

!!! tip "Choosing a model"
    - **Speech enhancement:** DeepFilterNet and MossFormer2 SE focus on cleanup and denoising.
    - **Source separation:** SAM-Audio is the text-guided separation option.
    - **Conversation / multimodal speech:** Moshi and Liquid2.5-Audio cover interactive audio workflows.
