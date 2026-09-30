#!/bin/bash
# patch_ctranslate2.sh — CTranslate2 CUDA wheel (Oli hardware only)
#
# The pip ctranslate2 package is CPU-only on aarch64.
# Oli has a locally-built CUDA wheel that must be force-reinstalled.
#
# Run after install.sh: bash patch_ctranslate2.sh

set -e

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv not found. Run install.sh first."
    exit 1
fi

source "$VENV/bin/activate"

echo "=== Patch: CTranslate2 CUDA wheel ==="
WHEEL=$(ls ~/CTranslate2/python/dist/ctranslate2-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "SKIP: ~/CTranslate2/python/dist/ctranslate2-*.whl not found."
    echo "Build CTranslate2 with CUDA support first, then re-run this script."
    exit 0
fi

pip uninstall -y ctranslate2
pip install --force-reinstall --no-deps "$WHEEL"
echo "Done: $(basename "$WHEEL")"
