<!-- AUTO-GENERATED: do not edit by hand. Run scripts/generate_model_catalog.py -->

# Models

MLX Audio supports a wide range of audio models across four categories, all optimized for Apple Silicon.

Many hosted MLX checkpoints referenced in these docs live under [mlx-community](https://huggingface.co/mlx-community) on Hugging Face, the shared org for ready-to-use MLX model weights across projects like `mlx-lm`, `mlx-vlm`, and `mlx-audio`. If you are adding a new model, prefer publishing it there when possible so users can find MLX models in one consistent place.

## Text-to-Speech

MLX Audio supports a wide range of TTS models optimized for Apple Silicon. The generated catalog below is derived from `ModelConfig.DOCS` metadata so model capabilities stay in sync with the implementation.

| Model | Description | Languages | Repo |
|-------|-------------|-----------|------|
| [**Bark**](tts/generated/bark.md) | Promptable multilingual TTS with preset voice prompts | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | -- |
| [**Chatterbox**](tts/generated/chatterbox.md) | Expressive multilingual TTS with voice cloning | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">ES</span><span class="model-language-chip">FR</span><span class="model-language-chip">DE</span><span class="model-language-chip">IT</span><span class="model-language-chip">PT</span><span class="model-language-chip">PL</span><span class="model-language-chip">TR</span><span class="model-language-chip">RU</span><span class="model-language-chip">NL</span><span class="model-language-chip">CS</span><span class="model-language-chip">AR</span><span class="model-language-chip">ZH</span><span class="model-language-chip">JA</span><span class="model-language-chip">HU</span><span class="model-language-chip">KO</span></div> | [mlx-community/chatterbox-fp16](https://huggingface.co/mlx-community/chatterbox-fp16) |
| [**Chatterbox Turbo**](tts/generated/chatterbox-turbo.md) | Low-latency expressive TTS with voice cloning and streaming | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) |
| [**CSM**](tts/generated/csm.md) | Conversational speech model with voice cloning | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | [mlx-community/csm-1b](https://huggingface.co/mlx-community/csm-1b) |
| [**Dia**](tts/generated/dia.md) | Dialogue-focused TTS with speaker tags | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | [mlx-community/Dia-1.6B-fp16](https://huggingface.co/mlx-community/Dia-1.6B-fp16) |
| [**Echo TTS**](tts/generated/echo-tts.md) | Diffusion-based TTS with fast voice cloning | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | [mlx-community/echo-tts-base](https://huggingface.co/mlx-community/echo-tts-base) |
| [**Fish Speech**](tts/generated/fish-speech.md) | Voice cloning and multi-speaker TTS with inline control tags | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/fish-audio-s2-pro](https://huggingface.co/mlx-community/fish-audio-s2-pro) |
| [**IndexTTS**](tts/generated/indextts.md) | Zero-shot TTS with voice cloning and BigVGAN decoding | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/IndexTTS](https://huggingface.co/mlx-community/IndexTTS) |
| [**Irodori TTS**](tts/generated/irodori-tts.md) | Japanese TTS with DiT and DACVAE decoding | <div class="model-language-chips"><span class="model-language-chip">JA</span></div> | [mlx-community/Irodori-TTS-500M-fp16](https://huggingface.co/mlx-community/Irodori-TTS-500M-fp16) |
| [**KittenTTS**](tts/generated/kitten-tts.md) | Small English TTS with preset expressive voices | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | -- |
| [**Kokoro**](tts/generated/kokoro.md) | Fast, high-quality multilingual TTS | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">JA</span><span class="model-language-chip">ZH</span><span class="model-language-chip">FR</span><span class="model-language-chip">ES</span><span class="model-language-chip">IT</span><span class="model-language-chip">PT</span><span class="model-language-chip">HI</span></div> | [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) |
| [**KugelAudio**](tts/generated/kugelaudio.md) | 7B multilingual TTS for European languages with diffusion decoding | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">DE</span><span class="model-language-chip">FR</span><span class="model-language-chip">ES</span><span class="model-language-chip">IT</span><span class="model-language-chip">PT</span><span class="model-language-chip">NL</span><span class="model-language-chip">PL</span><span class="model-language-chip">RU</span><span class="model-language-chip">UK</span><span class="model-language-chip">CS</span><span class="model-language-chip">RO</span><span class="model-language-chip">HU</span><span class="model-language-chip">SV</span><span class="model-language-chip">DA</span><span class="model-language-chip">FI</span><span class="model-language-chip">NO</span><span class="model-language-chip">EL</span><span class="model-language-chip">BG</span><span class="model-language-chip">SK</span><span class="model-language-chip">HR</span><span class="model-language-chip">SR</span><span class="model-language-chip">TR</span></div> | [kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open) |
| [**Ming Omni TTS (BailingMM)**](tts/generated/ming-omni-bailingmm.md) | Multimodal generation with voice cloning and style control | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">ZH</span></div> | [mlx-community/Ming-omni-tts-16.8B-A3B-bf16](https://huggingface.co/mlx-community/Ming-omni-tts-16.8B-A3B-bf16) |
| [**Ming Omni TTS (Dense)**](tts/generated/ming-omni-dense.md) | Lightweight Ming Omni variant for voice cloning and style control | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">ZH</span></div> | [mlx-community/Ming-omni-tts-0.5B-bf16](https://huggingface.co/mlx-community/Ming-omni-tts-0.5B-bf16) |
| [**OuteTTS**](tts/generated/outetts.md) | Efficient text-to-speech for Apple Silicon | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | [mlx-community/OuteTTS-1.0-0.6B-fp16](https://huggingface.co/mlx-community/OuteTTS-1.0-0.6B-fp16) |
| [**PocketTTS**](tts/generated/pocket-tts.md) | Compact streaming TTS with optional voice conditioning | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | -- |
| [**Qwen3-TTS**](tts/generated/qwen3-tts.md) | Multilingual TTS with voice cloning and voice design | <div class="model-language-chips"><span class="model-language-chip">ZH</span><span class="model-language-chip">EN</span><span class="model-language-chip">JA</span><span class="model-language-chip">KO</span><span class="model-language-chip">Multilingual</span></div> | [mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16) |
| [**Soprano**](tts/generated/soprano.md) | High-quality English TTS | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | [mlx-community/Soprano-1.1-80M-bf16](https://huggingface.co/mlx-community/Soprano-1.1-80M-bf16) |
| [**Spark**](tts/generated/spark-tts.md) | SparkTTS model for English and Chinese speech generation | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">ZH</span></div> | [mlx-community/Spark-TTS-0.5B-bf16](https://huggingface.co/mlx-community/Spark-TTS-0.5B-bf16) |
| [**TADA**](tts/generated/tada.md) | Text-Acoustic Dual Alignment TTS with voice cloning | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">Multilingual</span></div> | [HumeAI/mlx-tada-1b](https://huggingface.co/HumeAI/mlx-tada-1b) |
| [**VibeVoice**](tts/generated/vibevoice.md) | Streaming multilingual TTS with cached voice conditioning | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | -- |
| [**VoxCPM**](tts/generated/voxcpm.md) | Multilingual TTS with voice cloning and 44.1kHz audio output | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | -- |
| [**Voxtral TTS**](tts/generated/voxtral-tts.md) | Mistral's 4B multilingual TTS | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">FR</span><span class="model-language-chip">ES</span><span class="model-language-chip">DE</span><span class="model-language-chip">IT</span><span class="model-language-chip">PT</span><span class="model-language-chip">NL</span><span class="model-language-chip">AR</span><span class="model-language-chip">HI</span></div> | [mlx-community/Voxtral-4B-TTS-2603-mlx-bf16](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-bf16) |

[:octicons-arrow-right-24: Browse Text-to-Speech](tts/index.md)

---

## Speech-to-Text

MLX Audio provides speech-to-text models ranging from lightweight English-only recognizers to multilingual systems with streaming, timestamps, translation, and diarization support.

| Model | Description | Languages | Repo |
|-------|-------------|-----------|------|
| [**Canary**](stt/generated/canary.md) | Multilingual ASR with speech translation | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | -- |
| [**Cohere Transcribe**](stt/generated/cohere-transcribe.md) | Multilingual offline ASR with long-form chunking | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) |
| [**FireRedASR2**](stt/generated/fireredasr2.md) | Bilingual Chinese and English ASR | <div class="model-language-chips"><span class="model-language-chip">ZH</span><span class="model-language-chip">EN</span></div> | [mlx-community/FireRedASR2-AED-mlx](https://huggingface.co/mlx-community/FireRedASR2-AED-mlx) |
| [**GLM-ASR**](stt/generated/glm-asr.md) | Speech recognition with a Whisper encoder and GLM decoder | <div class="model-language-chips"><span class="model-language-chip">ZH</span><span class="model-language-chip">EN</span></div> | -- |
| [**Granite Speech**](stt/generated/granite-speech.md) | ASR and speech translation from IBM Granite | <div class="model-language-chips"><span class="model-language-chip">EN</span><span class="model-language-chip">FR</span><span class="model-language-chip">DE</span><span class="model-language-chip">ES</span><span class="model-language-chip">PT</span><span class="model-language-chip">JA</span></div> | [ibm-granite/granite-4.0-1b-speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) |
| [**LASR-CTC**](stt/generated/lasr-ctc.md) | CTC-based speech recognition with a LASR encoder | -- | -- |
| [**MMS**](stt/generated/mms.md) | Massively multilingual speech recognition | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) |
| [**Moonshine**](stt/generated/moonshine.md) | Lightweight English ASR | <div class="model-language-chips"><span class="model-language-chip">EN</span></div> | [UsefulSensors/moonshine-base](https://huggingface.co/UsefulSensors/moonshine-base) |
| [**Parakeet**](stt/generated/parakeet.md) | Fast multilingual ASR with streaming support | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |
| [**Qwen2-Audio**](stt/generated/qwen2-audio.md) | Audio-language model for transcription, translation, and audio understanding | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/Qwen2-Audio-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit) |
| [**Qwen3-ASR**](stt/generated/qwen3-asr.md) | Multilingual ASR with streaming support | <div class="model-language-chips"><span class="model-language-chip">ZH</span><span class="model-language-chip">EN</span><span class="model-language-chip">JA</span><span class="model-language-chip">KO</span><span class="model-language-chip">Multilingual</span></div> | [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) |
| [**Qwen3-ForcedAligner**](stt/generated/qwen3-forced-aligner.md) | Word-level forced alignment for speech transcripts | <div class="model-language-chips"><span class="model-language-chip">ZH</span><span class="model-language-chip">EN</span><span class="model-language-chip">JA</span><span class="model-language-chip">KO</span><span class="model-language-chip">Multilingual</span></div> | [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) |
| [**SenseVoice**](stt/generated/sensevoice.md) | Multilingual speech recognition with emotion and event detection | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/SenseVoiceSmall](https://huggingface.co/mlx-community/SenseVoiceSmall) |
| [**VibeVoice-ASR**](stt/generated/vibevoice-asr.md) | Multilingual ASR with diarization support | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/VibeVoice-ASR-bf16](https://huggingface.co/mlx-community/VibeVoice-ASR-bf16) |
| [**Voxtral**](stt/generated/voxtral.md) | Multilingual speech model from Mistral | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/Voxtral-Mini-3B-2507-bf16](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16) |
| [**Voxtral Realtime**](stt/generated/voxtral-realtime.md) | Streaming multilingual speech-to-text | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) |
| [**Whisper**](stt/generated/whisper.md) | OpenAI's robust STT model | <div class="model-language-chips"><span class="model-language-chip">Multilingual</span></div> | [mlx-community/whisper-large-v3-turbo-asr-fp16](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16) |

[:octicons-arrow-right-24: Browse Speech-to-Text](stt/index.md)

---

## Speech-to-Speech

MLX Audio includes speech enhancement, source separation, and multimodal speech interaction models under the STS umbrella.

| Model | Description | Repo |
|-------|-------------|------|
| [**DeepFilterNet**](sts/generated/deepfilternet.md) | Speech enhancement and noise suppression | [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx) |
| [**Liquid2.5-Audio**](sts/generated/lfm2-5-audio.md) | Multimodal speech-to-speech, TTS, and ASR in one model | [mlx-community/LFM2.5-Audio-1.5B-4bit](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-4bit) |
| [**Moshi**](sts/generated/moshi.md) | Full-duplex voice conversation model | [kyutai/moshiko-mlx-q4](https://huggingface.co/kyutai/moshiko-mlx-q4) |
| [**MossFormer2 SE**](sts/generated/mossformer2-se.md) | Speech enhancement and noise removal | [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX) |
| [**SAM-Audio**](sts/generated/sam-audio.md) | Text-guided audio source separation | [mlx-community/sam-audio-large](https://huggingface.co/mlx-community/sam-audio-large) |

[:octicons-arrow-right-24: Browse Speech-to-Speech](sts/index.md)

---

## Voice Activity Detection

MLX Audio includes endpoint detection and diarization models for conversational turn-taking and speaker-aware processing.

| Model | Description | Repo |
|-------|-------------|------|
| [**Smart Turn**](vad/generated/smart-turn.md) | Endpoint detection for conversational turn-taking | [mlx-community/smart-turn-v3](https://huggingface.co/mlx-community/smart-turn-v3) |
| [**Sortformer v1**](vad/generated/sortformer-v1.md) | End-to-end speaker diarization for up to four speakers | [mlx-community/diar_sortformer_4spk-v1-fp32](https://huggingface.co/mlx-community/diar_sortformer_4spk-v1-fp32) |
| [**Sortformer v2.1**](vad/generated/sortformer-v2-1.md) | Streaming speaker diarization with AOSC compression | [mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32) |

[:octicons-arrow-right-24: Browse Voice Activity Detection](vad/index.md)
