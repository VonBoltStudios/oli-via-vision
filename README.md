# Oli Voice Interaction App

A locally-running, end-to-end voice conversation system for the Oli humanoid robot. No cloud dependencies — all inference (speech-to-text, LLM, text-to-speech) runs on the Jetson Orin NX. Microphone audio comes in from, and speech goes out to, the Oli SDK over its WebSocket API (`ws://10.192.1.2:5000` by default).

## How it works

Oli operates in two modes:

**Background mode** — continuously listens for speech onset. When something is heard, it transcribes the clip and uses the LLM to determine whether someone is addressing Oli. Bystander conversation and ambient noise are ignored.

**Conversation mode** — once Oli's attention is triggered, it enters a full back-and-forth conversation. The conversation continues until the user signals they're done ("that's all", "goodbye", etc.) or a silence timeout elapses, at which point Oli returns to background listening.

### Voice commands

Explicit commands are matched against the transcript (case and punctuation ignored) and work in both modes:

| Command | Config key | Behavior |
|---|---|---|
| Shutdown | `shutdown_command` (default "shut down the voice service") | Exits the app. Works in any state, including muted. |
| Soft mute | `mute_command` (default "oli stop listening") | Oli says "Ok.", drops to background mode, and stops responding. Speech is still transcribed so commands can be heard. |
| Unmute | `unmute_command` (default "oli resume listening") | Oli says "Ready." and resumes normal wake detection. |
| Pre-recorded audio | `audio_commands` (list of `phrase` + `file`) | Plays a WAV via the SDK's `request_audio_play_file` instead of live TTS. Works in either mode, muted or not; does not change mute state and is not added to the LLM conversation history. |

`mute_command` and `unmute_command` accept a single phrase or a list (useful for Whisper spelling variants such as "Ollie"). Mute state is in memory only — a restart comes back unmuted.

For `audio_commands`, a bare filename resolves against the app root; use a full path (or URL) for anywhere else. Files must be reachable by the SDK backend (e.g. `/opt/limx/install/oli/asserts/startup.wav`).

```yaml
audio_commands:
  - phrase: "oli play intro"
    file: "intro.wav"
```

## Requirements

- Oli humanoid robot with its onboard NVIDIA Jetson Orin NX (Ubuntu 22.04, aarch64, JetPack 6.x), with the SDK reachable at `sdk.host`:`sdk.port`
- CUDA drivers (JetPack — should be pre-installed)
- Internet connection for first-time installation and model downloads only

## Installation

```bash
chmod +x install.sh start.sh
./install.sh
```

This will:
- Install Python dependencies (faster-whisper, piper-tts, ollama, websocket-client, and others) into `.venv`
- Download the default Piper TTS voice (`en_US-ryan-medium`) to `voices/`
- Install Ollama and pull the `qwen2.5:3b` model

### Oli hardware patches

The stock pip wheels for CTranslate2 (Whisper) and ONNX Runtime (Piper) are CPU-only on aarch64. After `install.sh`, run the patches to enable GPU inference:

```bash
bash patch_oli.sh              # runs both patches
# or individually:
bash patch_ctranslate2.sh      # CUDA CTranslate2 wheel (STT)
bash patch_onnxruntime.sh      # ONNX Runtime GPU wheel for JetPack 6.x / CUDA 12.6 (TTS)
```

Piper then runs with `CUDAExecutionProvider` (CPU fallback). Reference tests for the SDK audio paths are in `ws_stt_test.py`, `ws_tts_test.py`, and `llm_test.py`.

## Starting

```bash
./start.sh
```

Oli will announce it's ready and begin listening. Speak naturally — no specific trigger phrase required. Just address it by name or start talking to it.

To stop: say the shutdown command, or press `Ctrl+C`.

A different config file can be passed as an argument: `python3 main.py my-config.yaml`. Setting `push_to_talk: true` enables a dev mode (press Enter to speak); the voice commands above apply to the live listening path only.

## Configuration

All settings are in `config.yaml`. Key sections:

| Section | What it controls |
|---|---|
| `persona` | Oli's name, personality, system prompt (includes Oli's robot specifications) |
| `background` | Listening sensitivity, clip length, and wake detection timing |
| `conversation` | Silence timeout before returning to background |
| `shutdown_command` | Phrase that exits the app |
| `mute_command` / `unmute_command` | Soft mute / unmute phrases |
| `audio_commands` | Phrase → pre-recorded WAV playback |
| `sdk` | Oli SDK WebSocket host and port (mic input and speaker output) |
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
ollama rm qwen2.5:3b

# 3. Uninstall Ollama (optional)
sudo systemctl stop ollama
sudo rm /usr/local/bin/ollama
rm -rf ~/.ollama
```

## Troubleshooting

**No audio in or out:** Mic and speaker both go through the Oli SDK WebSocket. Confirm `sdk.host`/`sdk.port` in `config.yaml` and that the SDK is running. `ws_stt_test.py` and `ws_tts_test.py` exercise each path in isolation.

**Ollama not starting:** Run `ollama serve` manually in one terminal, then `./start.sh` in another.

**Response too slow:** Make sure the hardware patches were applied (otherwise STT/TTS run on CPU), or switch to a smaller STT model or LLM in `config.yaml`. The tested defaults are `medium` Whisper and `qwen2.5:3b`.

**Mute or audio command not triggering:** Whisper may spell "Oli" differently ("Ollie", "Ali"). Add variants by turning the phrase into a list in `config.yaml` (mute/unmute), or add another `audio_commands` entry.

**Oli triggering too easily in background:** Raise `background.silence_threshold_ms` (more speech required before a clip is classified) or adjust the `wake_classifier_prompt` to be more conservative.
