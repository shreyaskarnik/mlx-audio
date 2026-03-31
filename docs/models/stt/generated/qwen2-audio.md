<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->

# Qwen2-Audio

Audio-language model for transcription, translation, and audio understanding

| Field | Value |
|-------|-------|
| Task | Speech-to-Text |
| Languages | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> |
| Repo | [mlx-community/Qwen2-Audio-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit) |
| Source Docs | [Source Docs](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/stt/qwen2-audio.md) |
| Pipeline Tag | `automatic-speech-recognition` |
| Features | translation, audio understanding |

## Documentation

Qwen2-Audio is a multimodal audio-language model that handles more than transcription. In MLX Audio it can be used for ASR, translation, captioning, emotion recognition, and general audio understanding through text prompts.

## Models

| Model | Quantization | HuggingFace |
|-------|--------------|-------------|
| `Qwen/Qwen2-Audio-7B-Instruct` | bf16 | [:octicons-link-external-16: Model Card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| `mlx-community/Qwen2-Audio-7B-Instruct-4bit` | 4-bit | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit) |

## Usage

=== "CLI"

    ```bash
    python -m mlx_audio.stt.generate \
        --model mlx-community/Qwen2-Audio-7B-Instruct-4bit \
        --audio audio.wav \
        --prompt "Transcribe the audio."
    ```

=== "Python"

    ```python
    from mlx_audio.stt.utils import load_model

    model = load_model("mlx-community/Qwen2-Audio-7B-Instruct-4bit")

    result = model.generate("audio.wav", prompt="Transcribe the audio.")
    print(result.text)
    ```

## Typical Prompts

- `Transcribe the audio.`
- `Translate the speech to French.`
- `What emotion is the speaker expressing?`
- `Describe the environmental sounds in this clip.`

## Capabilities

- Speech transcription
- Speech translation
- Audio captioning
- Emotion and sentiment analysis
- Environmental sound classification
- General audio understanding

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/stt/models/qwen2_audio)
- [:octicons-mark-github-16: In-repo README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen2_audio/README.md)
- [:octicons-link-external-16: Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
