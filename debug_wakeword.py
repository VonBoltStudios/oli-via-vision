#!/usr/bin/env python3
"""Debug script: prints mic levels and wake word scores in real-time."""

import sys
import yaml
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

config_path = sys.argv[1] if len(sys.argv) > 1 else "config.mac.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

sample_rate = cfg["audio"]["sample_rate"]
chunk_ms = cfg["audio"]["chunk_ms"]
chunk_frames = int(sample_rate * chunk_ms / 1000)
phrase = cfg["wake_word"]["phrase"]
threshold = cfg["wake_word"]["threshold"]

print(f"Loading wake word model: {phrase}")
model = Model(wakeword_models=[phrase], inference_framework="onnx")
print(f"Listening... speak into your mic. Threshold: {threshold}")
print("(Ctrl+C to stop)\n")

with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=chunk_frames) as stream:
    while True:
        chunk, _ = stream.read(chunk_frames)
        audio = chunk.flatten()
        level = np.abs(audio).mean()
        pred = model.predict(audio)
        score = pred.get(phrase, 0.0)
        bar = "█" * int(score * 40)
        print(f"  mic: {level:.4f}  |  {phrase}: {score:.3f}  {bar}", end="\r")
        if score >= threshold:
            print(f"\n*** WAKE WORD DETECTED (score: {score:.3f}) ***")
