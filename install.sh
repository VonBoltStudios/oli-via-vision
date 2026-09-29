#!/bin/bash
# Oli Voice Interaction App — installation script
# Target: NVIDIA Jetson Orin NX, Ubuntu 22.04, aarch64

set -e

echo "=== Oli VIA Installer ==="
echo "Platform: $(uname -m) | $(lsb_release -ds 2>/dev/null || echo 'unknown')"
echo ""

# --- System dependencies ---
echo "[1/4] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y -q \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libsndfile1 \
    wget \
    curl

# --- Python virtual environment ---
echo "[2/4] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q
pip install -q \
    faster-whisper \
    piper-tts \
    ollama \
    websocket-client \
    sounddevice \
    soundfile \
    pyyaml \
    numpy

# --- Piper voice download ---
echo "[3/4] Downloading Piper TTS voice..."
VOICE_DIR="$(pwd)/voices"
mkdir -p "$VOICE_DIR"
VOICE="en_US-ryan-medium"
if [ ! -f "$VOICE_DIR/${VOICE}.onnx" ]; then
    echo "  Downloading: $VOICE"
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/${VOICE}.onnx" \
        -O "$VOICE_DIR/${VOICE}.onnx"
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/${VOICE}.onnx.json" \
        -O "$VOICE_DIR/${VOICE}.onnx.json"
    echo "  Voice saved to $VOICE_DIR/"
else
    echo "  Voice already present — skipping."
fi

# --- Ollama ---
echo "[4/4] Installing Ollama and pulling LLM..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
fi

echo "  Starting Ollama service..."
ollama serve &>/dev/null &
sleep 3

echo "  Pulling qwen2.5:3b (this may take a few minutes)..."
ollama pull qwen2.5:3b

# --- Done ---
echo ""
echo "=== Installation complete ==="
echo ""
echo "To start Oli:   ./start.sh"
echo "To configure:   edit config.yaml"
echo ""
echo "Voice files are in: voices/"
echo "To use a different voice, download the .onnx and .onnx.json files"
echo "from https://huggingface.co/rhasspy/piper-voices and update tts.voice in config.yaml"
