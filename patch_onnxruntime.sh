#!/bin/bash
# patch_onnxruntime.sh — ONNX Runtime GPU wheel (Oli hardware only)
#
# The standard onnxruntime pip package is CPU-only on aarch64.
# Replaces it with NVIDIA's official Jetson GPU wheel (JP6 / CUDA 12.6 / cuDNN 9.x)
# to enable CUDA inference for Piper TTS.
#
# Source: https://pypi.jetson-ai-lab.io/jp6/cu126
# Verified: JP 6.2.1, Python 3.10, cuDNN 9.x (libcudnn.so.9)
#
# Run after install.sh: bash patch_onnxruntime.sh

set -e

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv not found. Run install.sh first."
    exit 1
fi

source "$VENV/bin/activate"

echo "=== Patch: ONNX Runtime GPU wheel ==="
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
echo "Done."
echo ""
echo "Verify (expect CUDAExecutionProvider and TensorrtExecutionProvider):"
echo "  python3 -c \"import onnxruntime; print(onnxruntime.get_available_providers())\""
