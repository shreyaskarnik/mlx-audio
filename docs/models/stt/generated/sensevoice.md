<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->

# SenseVoice

Multilingual speech recognition with emotion and event detection

| Field | Value |
|-------|-------|
| Task | Speech-to-Text |
| Languages | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> |
| Repo | [mlx-community/SenseVoiceSmall](https://huggingface.co/mlx-community/SenseVoiceSmall) |
| Source Docs | [Source Docs](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/README.md) |
| Pipeline Tag | `automatic-speech-recognition` |
| Features | emotion detection, event detection |

## Documentation

MLX implementation of [SenseVoice Small](https://github.com/FunAudioLLM/SenseVoice) from Alibaba DAMO Academy. Non-autoregressive CTC-based model with speech recognition, language identification, emotion recognition, and audio event detection.

## Features

- 50+ languages (strong on zh, en, ja, ko, yue)
- emotion detection (happy, sad, angry, neutral, etc.)
- audio event detection (speech, bgm, laughter, applause)
- non-autoregressive inference (very fast, ~70ms for 10s audio)
- ~234M parameters

## Usage

```python
from mlx_audio.stt import load

# downloads automatically from huggingface
model = load("mlx-community/SenseVoiceSmall")

result = model.generate("audio.wav", language="auto")
print(result.text)
print(result.language)

# rich info (emotion, event) is in the first segment
seg = result.segments[0]
print(seg["emotion"])  # e.g. "neutral"
print(seg["event"])    # e.g. "Speech"
```

### Language options

`"auto"`, `"zh"`, `"en"`, `"ja"`, `"ko"`, `"yue"`, `"nospeech"`

### Inverse text normalization

```python
result = model.generate("audio.wav", language="auto", use_itn=True)
```
