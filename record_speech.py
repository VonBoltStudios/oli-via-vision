#!/usr/bin/env python3
"""Record prepared statements as WAV files using the same Piper TTS voice as Oli.

Usage: python3 record_speech.py [config.yaml]

Inline [p0.6] in the text inserts 0.6 s of silence; tts.sentence_pause_s (config, default
0.35) sets the gap between sentences.

Files are saved to recordings/ (gitignored). Play one back with the audio_commands
config, e.g.  file: "recordings/intro.wav"  (relative paths resolve against the app root).
"""

import os
import re
import sys
import threading
import time
import wave
from datetime import datetime

import numpy as np
import yaml

from modules.tts import TTS

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(APP_ROOT, "recordings")


def safe_name(title: str) -> str:
    name = re.sub(r"[^\w\- ]", "", title).strip().replace(" ", "_")
    return name or datetime.now().strftime("TTS_%y-%m-%d_%H%M%S")


def read_text() -> str:
    """Typed/pasted text (finish with a blank line), or a path to a plain text file."""
    print("\nText to record — type or paste (finish with a blank line), or give a .txt file path:")
    first = input("> ").strip().strip("\"'")
    path = os.path.expanduser(first)
    if first and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        print(f"  [loaded {len(text)} characters from {path}]")
        return text
    lines = [first] if first else []
    while True:
        line = input("> " if not lines else "  ").rstrip()
        if not line:
            break
        lines.append(line)
    return " ".join(lines).strip()


def with_spinner(label: str, fn):
    """Run fn() while showing a simple elapsed-time indicator."""
    done = threading.Event()
    start = time.monotonic()

    def spin():
        frames = "|/-\\"
        i = 0
        while not done.wait(0.15):
            print(f"\r  {frames[i % 4]} {label} {time.monotonic() - start:4.1f}s", end="", flush=True)
            i += 1

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        done.set()
        t.join()
        print("\r" + " " * (len(label) + 14) + "\r", end="")


PAUSE_TAG = re.compile(r"\[p(\d+(?:\.\d+)?)\]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def synthesize_with_pauses(tts, text: str, sentence_pause_s: float = 0.0):
    """Synthesize text one sentence at a time, joined with explicit silence.

    Inline [p0.6] inserts 0.6 s of silence at that point. sentence_pause_s adds a default
    gap between sentences. Line breaks/extra whitespace are collapsed (they upset Piper).
    """
    def silence(seconds):
        return np.zeros(int(tts.sample_rate * seconds), dtype=np.int16)

    parts, pending, total_pause = [], 0.0, 0.0
    pieces = PAUSE_TAG.split(text)  # text, pause, text, pause, ...
    for i, piece in enumerate(pieces):
        if i % 2:  # pause tag
            pending += float(piece)
            continue
        for sentence in SENTENCE_END.split(" ".join(piece.split())):
            if not sentence:
                continue
            gap = pending + (sentence_pause_s if parts else 0.0)
            if gap:
                parts.append(silence(gap))
                total_pause += gap
            pending = 0.0
            seg = tts.synthesize(sentence)
            print(f"  [{len(seg) / tts.sample_rate:4.1f}s] {sentence[:60]}")
            parts.append(seg)
    if pending:  # trailing tag
        parts.append(silence(pending))
        total_pause += pending
    print(f"  ({total_pause:.1f}s of pauses)")
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)


def write_wav(path: str, pcm, sample_rate: int):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tts = TTS(
        voice=cfg["tts"]["voice"],
        speed=cfg["tts"]["speed"],
        sdk_host=cfg["sdk"]["host"],
        sdk_port=cfg["sdk"]["port"],
        connect=False,  # SDK connection only needed for preview
    )
    os.makedirs(REC_DIR, exist_ok=True)

    while True:
        title = input("\nRecording title (blank for TTS_YY-MM-DD_hhmmss): ").strip()
        name = safe_name(title) if title else datetime.now().strftime("TTS_%y-%m-%d_%H%M%S")
        path = os.path.join(REC_DIR, name + ".wav")
        text = read_text()
        if not text:
            print("  [no text — skipped]")
            continue

        while True:
            # Spinner would overwrite the per-segment lines, so print them plainly.
            pcm = synthesize_with_pauses(tts, text, cfg["tts"].get("sentence_pause_s", 0.35))
            if len(pcm) == 0:
                print("  [nothing generated]")
            else:
                write_wav(path, pcm, tts.sample_rate)
                print(f"  Saved {path}  ({len(pcm) / tts.sample_rate:.1f}s, {tts.sample_rate} Hz mono)")

            while True:
                choice = input("[p]review on Oli, [r]egenerate, [n]ew prompt, [e]xit: ").strip().lower()[:1]
                if choice == "p":
                    if not os.path.isfile(path):
                        print("  [no file to preview]")
                        continue
                    try:
                        tts.ensure_connected()
                        with_spinner("Playing", lambda: tts.play_file(path))
                    except Exception as e:
                        print(f"  [preview failed: {e}]")
                elif choice in ("r", "n", "e"):
                    break
            if choice == "r":
                continue
            break
        if choice == "e":
            break
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
