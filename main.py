#!/usr/bin/env python3
"""Oli Voice Interaction App — background/conversation state machine."""

import sys
import yaml
from modules.audio import WSAudioCapture
from modules.stt import STT
from modules.llm import LLM
from modules.tts import TTS


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)

    audio = WSAudioCapture(
        host=cfg["sdk"]["host"],
        port=cfg["sdk"]["port"],
        sample_rate=cfg["audio"]["sample_rate"],
        channels=cfg["audio"]["channels"],
        chunk_ms=cfg["audio"]["chunk_ms"],
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
        sdk_host=cfg["sdk"]["host"],
        sdk_port=cfg["sdk"]["port"],
    )

    name = cfg["persona"]["name"]
    ptt_mode = cfg.get("push_to_talk", False)
    shutdown_cmd = cfg.get("shutdown_command", "shut down the voice service").lower()

    bg = cfg.get("background", {})
    bg_wait_s = bg.get("wait_timeout_s", 0.5)
    bg_silence_ms = bg.get("silence_threshold_ms", 800)
    bg_max_clip_s = bg.get("max_clip_s", 3)
    wake_prompt_template = bg.get("wake_classifier_prompt", (
        f'Someone said the following near a robot named {name}. '
        f'Are they trying to get {name}\'s attention or start a conversation? '
        f'Reply YES or NO only.\n\nThey said: "{{text}}"'
    ))

    conv = cfg.get("conversation", {})
    conv_wait_s = conv.get("wait_timeout_s", 8)
    conv_silence_ms = cfg["audio"]["silence_threshold_ms"]
    conv_max_s = cfg["audio"]["max_record_s"]

    print(f"\n{name} is ready.")
    if ptt_mode:
        print("Mode: push-to-talk (dev) — press Enter to speak, Ctrl+C to quit.\n")
        tts.speak(f"Hello, I'm {name}. Press Enter whenever you want to speak.")
    else:
        print("Mode: background listening — speak naturally to start a conversation.\n")
        tts.speak(f"Hello, I'm {name}. Just speak to me whenever you're ready.")

    while True:
        # ── BACKGROUND MODE ───────────────────────────────────────────────────
        if ptt_mode:
            input("\n  [press Enter to speak]")
            awakened = True
        else:
            awakened = False
            print("  [background — listening...]", end="\r")
            while not awakened:
                clip = audio.wait_and_record(
                    wait_timeout_s=bg_wait_s,
                    silence_threshold_ms=bg_silence_ms,
                    max_record_s=bg_max_clip_s,
                )
                if clip is None:
                    continue  # no speech onset in this window

                text = stt.transcribe(clip, sample_rate=cfg["audio"]["sample_rate"])
                if not text:
                    continue

                if shutdown_cmd in text.lower():
                    print(f"\n  [shutdown command]")
                    tts.speak("Shutting down.")
                    sys.exit(0)

                answer = llm.classify(wake_prompt_template.replace("{text}", text))
                if "YES" in answer.upper():
                    print(f"\n  [wake — heard: {text}]")
                    awakened = True
                else:
                    print(f"  [background] ignored: {text[:60]!r}", end="\r")

        # ── CONVERSATION MODE ─────────────────────────────────────────────────
        tts.speak("Yes?")
        llm.reset()

        while True:
            if ptt_mode:
                input("  [press Enter to speak]")
                utterance = audio.record_until_silence(
                    silence_threshold_ms=conv_silence_ms,
                    max_record_s=conv_max_s,
                )
            else:
                utterance = audio.wait_and_record(
                    wait_timeout_s=conv_wait_s,
                    silence_threshold_ms=conv_silence_ms,
                    max_record_s=conv_max_s,
                )
                if utterance is None:
                    print("\n  [conversation timeout — returning to background]")
                    tts.speak("I'll be here if you need me.")
                    break

            text = stt.transcribe(utterance, sample_rate=cfg["audio"]["sample_rate"])
            if not text:
                continue

            print(f"  You: {text}")

            if shutdown_cmd in text.lower():
                tts.speak("Shutting down.")
                sys.exit(0)

            exit_prompt = (
                f'The user said: "{text}"\n'
                f'Are they signaling the end of the conversation — saying goodbye, '
                f'done, that\'s all, thanks, see you, etc.? Reply YES or NO only.'
            )
            is_ending = "YES" in llm.classify(exit_prompt).upper()

            reply = llm.chat(text)
            print(f"  {name}: {reply}")
            tts.speak(reply)

            if is_ending:
                print("  [conversation ended]")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down.")
        sys.exit(0)
