<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->

# Parakeet

Fast multilingual ASR with streaming support

| Field | Value |
|-------|-------|
| Task | Speech-to-Text |
| Languages | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> |
| Repo | [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |
| Source Docs | [Source Docs](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/stt/parakeet.md) |
| Pipeline Tag | `automatic-speech-recognition` |
| Features | Streaming, Timestamps, word timestamps, streaming |

## Documentation

[Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) is NVIDIA's high-accuracy speech-to-text model. Parakeet v2 focuses on English, while Parakeet v3 supports 25 European languages.

## Available Models

| Model | Languages | Description | Repo |
|-------|-----------|-------------|------|
| **Parakeet v2** | English | English-only, high accuracy | [mlx-community/parakeet-tdt-0.6b-v2](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) |
| **Parakeet v3** | 25 EU languages | Multilingual European | [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |

## Supported Languages (v3)

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian

## Python Usage

### Basic Transcription

```python
from mlx_audio.stt.utils import load

# Load the multilingual v3 model
model = load("mlx-community/parakeet-tdt-0.6b-v3")

# Transcribe audio
result = model.generate("audio.wav")
print(f"Text: {result.text}")
```

### Sentence and Word Timestamps

```python
result = model.generate("audio.wav")
for sentence in result.sentences:
    print(f"[{sentence.start:.2f}s - {sentence.end:.2f}s] {sentence.text}")
```

### Streaming Transcription

```python
for chunk in model.generate("long_audio.wav", stream=True):
    print(chunk.text, end="", flush=True)
```

## CLI Usage

=== "Basic transcription"

    ```bash
    python -m mlx_audio.stt.generate \
      --model mlx-community/parakeet-tdt-0.6b-v3 \
      --audio speech.wav \
      --output-path output \
      --verbose
    ```

=== "JSON output"

    ```bash
    python -m mlx_audio.stt.generate \
      --model mlx-community/parakeet-tdt-0.6b-v3 \
      --audio speech.wav \
      --output-path output \
      --format json \
      --verbose
    ```

!!! tip "v2 vs v3"
    Use **v2** for English-only workloads where you want the best English accuracy. Use **v3** when you need multilingual European language support.
