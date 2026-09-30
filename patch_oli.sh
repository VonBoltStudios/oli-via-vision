#!/bin/bash
# patch_oli.sh — Oli hardware-specific patches
#
# Run this after install.sh on the target Oli robot.
# Applies hardware-specific package overrides that are not portable
# to generic Linux/Mac environments.
#
# Usage: bash patch_oli.sh

set -e

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv not found. Run install.sh first."
    exit 1
fi

source "$VENV/bin/activate"

echo "=== Oli Hardware Patches ==="
echo ""

# ── Patch 1: CTranslate2 CUDA wheel ──────────────────────────────────────────
# The pip ctranslate2 package is CPU-only on aarch64.
# Oli has a locally-built CUDA wheel that must be force-reinstalled.
echo "[1/2] Patching CTranslate2 with CUDA wheel..."
WHEEL=$(ls ~/CTranslate2/python/dist/ctranslate2-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "  SKIP: ~/CTranslate2/python/dist/ctranslate2-*.whl not found."
    echo "  Build CTranslate2 with CUDA support first, then re-run this script."
else
    pip uninstall -y ctranslate2
    pip install --force-reinstall --no-deps "$WHEEL"
    echo "  Done: $(basename "$WHEEL")"
fi

echo ""

# ── Patch 2: ONNX Runtime GPU wheel (Piper TTS) ──────────────────────────────
# The standard onnxruntime pip package is CPU-only on aarch64.
# Replace with NVIDIA's Jetson GPU wheel to enable CUDA inference for Piper.
#
# To find the right wheel:
#   1. Check JetPack version:  dpkg -l | grep jetpack
#   2. Find matching wheel at: https://elinux.org/Jetson_Zoo (ONNX Runtime section)
#   3. Set ONNXRUNTIME_WHEEL below to the download URL for your JetPack version.
#
echo "[2/2] Patching ONNX Runtime with Jetson GPU wheel..."
ONNXRUNTIME_WHEEL="https://nvidia.box.com/shared/static/6l0u97rj80ifwkk8rqbzj1try89fk26z.whl"  # onnxruntime 1.19.0, JetPack 6.0 (L4T r36.x), Python 3.10

if [ -z "$ONNXRUNTIME_WHEEL" ]; then
    echo "  SKIP: ONNXRUNTIME_WHEEL not set."
    echo "  Check JetPack version (dpkg -l | grep jetpack), find the matching"
    echo "  wheel at https://elinux.org/Jetson_Zoo, set ONNXRUNTIME_WHEEL in"
    echo "  this script, and re-run."
else
    # nvidia.box.com URLs redirect to a hash-named file — pip can't determine
    # the wheel filename from the URL alone, so download it first with a real name.
    WHEEL_FILE="/tmp/onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl"
    echo "  Downloading wheel..."
    wget -q -O "$WHEEL_FILE" "$ONNXRUNTIME_WHEEL"
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    pip install --force-reinstall --no-deps "$WHEEL_FILE"
    rm -f "$WHEEL_FILE"
    echo "  Done."
    echo "  Verify: python3 -c \"import onnxruntime; print(onnxruntime.get_available_providers())\""
fi

echo ""
echo "=== Patch complete ==="
