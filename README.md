# Oli Voice Interaction App

A locally-running, end-to-end voice conversation system for the Oli humanoid robot. No cloud dependencies — all inference runs on the Jetson Orin NX.

## How it works

Oli operates in two modes:

**Background mode** — continuously listens for speech onset. When something is heard, it transcribes the clip and uses the LLM to determine whether someone is addressing Oli. Bystander conversation and ambient noise are ignored.

**Conversation mode** — once Oli's attention is triggered, it enters a full back-and-forth conversation. The conversation continues until the user signals they're done ("that's all", "goodbye", etc.) or a silence timeout elapses, at which point Oli returns to background listening.

Saying the configured `shutdown_command` (default: "shut down the voice service") exits the app entirely.

## Requirements

- NVIDIA Jetson Orin NX, Ubuntu 22.04, aarch64
- CUDA drivers (JetPack — should be pre-installed)
- Internet connection for first-time model downloads only

## Installation

```bash
chmod +x install.sh start.sh
./install.sh
```

This will:
- Install Python dependencies (faster-whisper, piper-tts, ollama, sounddevice)
- Download the default Piper TTS voice to `voices/`
- Install Ollama and pull the `llama3.1:8b` model (~4.5 GB — takes a few minutes)

## Starting

```bash
./start.sh
```

Oli will announce it's ready and begin listening. Speak naturally — no specific trigger phrase required. Just address it by name or start talking to it.

Press `Ctrl+C` to stop.

## Configuration

All settings are in `config.yaml`. Key sections:

| Section | What it controls |
|---|---|
| `persona` | Oli's name, personality, and system prompt |
| `background` | Listening sensitivity, clip length, and wake detection timing |
| `conversation` | Silence timeout before returning to background |
| `shutdown_command` | Exact phrase that exits the app |
| `stt` | Whisper model size and compute settings |
| `llm` | Ollama model, temperature, response length |
| `tts` | Voice file path, speed |
| `audio` | Sample rate, silence detection thresholds |

No code changes needed — edit `config.yaml` and restart.

### Changing the TTS voice

Voice files are not bundled — they must be downloaded separately. Browse available voices at:

https://huggingface.co/rhasspy/piper-voices

Download the `.onnx` and `.onnx.json` files for your chosen voice into the `voices/` directory, then update `tts.voice` in `config.yaml` with the new path.

## Uninstalling

```bash
# 1. Remove the project directory (includes .venv and voice files)
rm -rf /path/to/oli-via

# 2. Remove downloaded Ollama models
ollama rm llama3.1:8b

# 3. Uninstall Ollama (optional)
sudo systemctl stop ollama
sudo rm /usr/local/bin/ollama
rm -rf ~/.ollama
```

## Troubleshooting

**Mic not working:** Check available devices:
```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```
Set `audio.input_device` in `config.yaml` to the correct device index if needed.

**Ollama not starting:** Run `ollama serve` manually in one terminal, then `./start.sh` in another.

**Response too slow:** Switch to a smaller STT model (`medium` instead of `large-v3`) or a smaller LLM in `config.yaml`.

**Oli triggering too easily in background:** Raise `background.silence_threshold_ms` (more speech required before a clip is classified) or adjust the `wake_classifier_prompt` to be more conservative.
