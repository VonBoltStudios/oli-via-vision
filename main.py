#!/usr/bin/env python3
"""Oli Voice Interaction App — main conversation loop."""

import sys
import yaml
from modules.audio import AudioCapture
from modules.wake_word import WakeWordDetector
from modules.stt import STT
from modules.llm import LLM
from modules.tts import TTS


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)

    audio = AudioCapture(
        sample_rate=cfg["audio"]["sample_rate"],
        channels=cfg["audio"]["channels"],
        chunk_ms=cfg["audio"]["chunk_ms"],
    )
    wake = WakeWordDetector(
        phrase=cfg["wake_word"]["phrase"],
        threshold=cfg["wake_word"]["threshold"],
    )
    stt = STT(
        model_size=cfg["stt"]["model"],
        language=cfg["stt"]["language"],
        device=cfg["stt"]["device"],
        compute_type=cfg["stt"]["compute_type"],
    )
    llm = LLM(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
        system_prompt=cfg["persona"]["system_prompt"],
    )
    tts = TTS(
        voice=cfg["tts"]["voice"],
        speed=cfg["tts"]["speed"],
        output_device=cfg["tts"].get("output_device"),
        piper_bin=cfg["tts"].get("piper_bin"),
    )

    name = cfg["persona"]["name"]
    wake_phrase = cfg["wake_word"]["phrase"]
    ptt_mode = cfg.get("push_to_talk", False)

    print(f"\n{name} is ready.")
    if ptt_mode:
        print("Mode: push-to-talk — press Enter to speak, Ctrl+C to quit.\n")
        tts.speak(f"Hello, I'm {name}. Press Enter whenever you want to speak.")
    else:
        print(f"Mode: wake word — say '{wake_phrase.replace('_', ' ')}' to speak.\n")
        tts.speak(f"Hello, I'm {name}. Say {wake_phrase.replace('_', ' ')} to talk to me.")

    while True:
        if ptt_mode:
            input("  [press Enter to speak]")
            tts.speak("Yes?")
        else:
            triggered = False
            for audio_chunk in audio.stream():
                if wake.check(audio_chunk):
                    triggered = True
                    break
            if not triggered:
                continue
            print("[wake word detected]")
            tts.speak("Yes, I'm here.")

        utterance_audio = audio.record_until_silence(
            silence_threshold_ms=cfg["audio"]["silence_threshold_ms"],
            max_record_s=cfg["audio"]["max_record_s"],
        )

        text = stt.transcribe(utterance_audio, sample_rate=cfg["audio"]["sample_rate"])
        if not text:
            print("  [no speech detected]")
            continue

        print(f"  You: {text}")

        reply = llm.chat(text)
        print(f"  {name}: {reply}")

        tts.speak(reply)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down.")
        sys.exit(0)
