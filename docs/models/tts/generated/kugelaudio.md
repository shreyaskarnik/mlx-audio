<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->

# KugelAudio

7B multilingual TTS for European languages with diffusion decoding

| Field | Value |
|-------|-------|
| Task | Text-to-Speech |
| Languages | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">DE</span><span class="model-language-chip">FR</span><span class="model-language-chip">ES</span><span class="model-language-chip">IT</span><span class="model-language-chip">PT</span><span class="model-language-chip">NL</span><span class="model-language-chip">PL</span><span class="model-language-chip">RU</span><span class="model-language-chip">UK</span><span class="model-language-chip">CS</span><span class="model-language-chip">RO</span><span class="model-language-chip">HU</span><span class="model-language-chip">SV</span><span class="model-language-chip">DA</span><span class="model-language-chip">FI</span><span class="model-language-chip">NO</span><span class="model-language-chip">EL</span><span class="model-language-chip">BG</span><span class="model-language-chip">SK</span><span class="model-language-chip">HR</span><span class="model-language-chip">SR</span><span class="model-language-chip">TR</span></div> |
| Repo | [kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open) |
| Source Docs | [Source Docs](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/tts/kugelaudio.md) |
| License | `mit` |
| Pipeline Tag | `text-to-speech` |

## Documentation

KugelAudio is an open-weight 7B text-to-speech model for 24 European languages. The MLX Audio integration runs the original weights directly and exposes the standard `generate()` interface.

## Model

| Model | Precision | HuggingFace |
|-------|-----------|-------------|
| `kugelaudio/kugelaudio-0-open` | bfloat16 | [:octicons-link-external-16: Model Card](https://huggingface.co/kugelaudio/kugelaudio-0-open) |

## Usage

=== "CLI"

    ```bash
    python -m mlx_audio.tts.generate \
        --model kugelaudio/kugelaudio-0-open \
        --text "Hello, this is KugelAudio." \
        --cfg_scale 3.0 \
        --ddpm_steps 10
    ```

=== "Python"

    ```python
    from mlx_audio.tts import load

    model = load("kugelaudio/kugelaudio-0-open")

    result = next(
        model.generate(
            text="Hello, this is KugelAudio running on Apple Silicon.",
            cfg_scale=3.0,
            ddpm_steps=10,
        )
    )
    audio = result.audio
    ```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cfg_scale` | `3.0` | Classifier-free guidance strength |
| `ddpm_steps` | `10` | Diffusion steps for quality vs speed |
| `max_tokens` | `2048` | Maximum speech tokens to generate |

## Languages

English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Ukrainian, Czech, Romanian, Hungarian, Swedish, Danish, Finnish, Norwegian, Greek, Bulgarian, Slovak, Croatian, Serbian, and Turkish.

## Notes

- KugelAudio uses a hybrid autoregressive plus diffusion pipeline derived from VibeVoice.
- The upstream release does not ship voice presets; generation uses the default voice path.
- Expect roughly 17 GB of unified memory for the full bfloat16 weights.

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/kugelaudio)
- [:octicons-mark-github-16: In-repo README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/kugelaudio/README.md)
- [:octicons-link-external-16: kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open)
