#!/bin/bash
# patch_oli.sh — Oli hardware-specific patches (runs all patches)
#
# Runs each patch script in sequence. You can also run them individually:
#   bash patch_ctranslate2.sh   — CTranslate2 CUDA wheel (Whisper / STT)
#   bash patch_onnxruntime.sh   — ONNX Runtime GPU wheel (Piper TTS)
#
# Usage: bash patch_oli.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Oli Hardware Patches ==="
echo ""

echo "[1/2] CTranslate2..."
bash "$SCRIPT_DIR/patch_ctranslate2.sh"

echo ""
echo "[2/2] ONNX Runtime..."
bash "$SCRIPT_DIR/patch_onnxruntime.sh"

echo ""
echo "=== All patches complete ==="
