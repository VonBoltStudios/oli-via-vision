"""Vision for Oli VIA: on-demand single-frame object detection -> scene text for the LLM.

The voice loop uses only this facade:

    vision = Vision(cfg["vision"]); vision.warmup(); vision.prepare_acks(tts)
    if vision.needs_vision(transcript):
        ack = vision.start_ack(tts)          # speaks "Let me take a look..." on a thread
        scene = vision.scene_snapshot()      # grab -> detect -> serialize (blocking)
        ack.join()
        prompt = vision.build_prompt(scene.text, transcript)
"""

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .camera import Frame, HeadCamera
from .detector import build_detector
from .gate import VisionGate
from .scene import serialize

DEFAULT_ACKS = [
    "Okay, let me take a look.",
    "One moment, let me look around.",
    "Sure, let me see what's in front of me.",
]
DEFAULT_FAIL = "I'm having trouble seeing right now."
DEFAULT_GROUNDING = (
    "Answer using only what your camera sees above. If it does not show what the person "
    "asks about, say you do not see it. Left and right are from your own point of view. "
    "Do not mention this note."
)


@dataclass
class SceneResult:
    ok: bool
    text: Optional[str] = None
    error: Optional[str] = None
    n_detections: int = 0
    timings: Dict[str, float] = field(default_factory=dict)
    frame: Optional[Frame] = None

    def timing_line(self) -> str:
        order = ["grab", "preprocess", "infer", "post", "serialize", "total"]
        return " | ".join(f"{k} {self.timings[k]:.3f}s" for k in order if k in self.timings)


class Vision:
    def __init__(self, cfg: dict):
        ccfg = cfg.get("camera") or {}
        self.camera = HeadCamera(
            serial=ccfg["serial"],
            width=ccfg.get("width", 1280),
            height=ccfg.get("height", 720),
            fps=ccfg.get("fps", 30),
            warmup_frames=ccfg.get("warmup_frames", 10),
            timeout_ms=ccfg.get("timeout_ms", 3000),
            depth=ccfg.get("depth", False),
        )
        self.detector = build_detector(cfg.get("detector") or {})
        self.gate = VisionGate(cfg.get("patterns"))
        self.max_objects = int(cfg.get("max_objects", 10))
        self.ack_phrases = cfg.get("ack_phrases") or DEFAULT_ACKS
        self.fail_phrase = cfg.get("fail_phrase", DEFAULT_FAIL)
        self.grounding = cfg.get("grounding", DEFAULT_GROUNDING)
        self._acks = []

    # ── startup ───────────────────────────────────────────────────────────────

    def warmup(self):
        """Load the detector and run one dummy inference so the first real one is fast."""
        import numpy as np

        t0 = time.monotonic()
        self.detector.load()
        self.detector.detect(np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8))
        print(f"[Vision] detector ready in {time.monotonic() - t0:.1f}s")

    def prepare_acks(self, tts):
        """Pre-synthesize the acknowledgement phrases so playback starts instantly."""
        self._acks = []
        for phrase in self.ack_phrases:
            if len(phrase.split()) < 4:
                print(f"[Vision] warning: ack {phrase!r} is very short; the SDK speaker may clip it")
            self._acks.append(tts.synthesize(phrase))
        print(f"[Vision] {len(self._acks)} ack phrase(s) ready")

    # ── per-turn ──────────────────────────────────────────────────────────────

    def needs_vision(self, transcript: str) -> bool:
        pattern = self.gate.match(transcript)
        print(f"  [vision gate] {'match ' + repr(pattern) if pattern else 'no match'}")
        return pattern is not None

    def start_ack(self, tts) -> Optional[threading.Thread]:
        """Speak a random pre-synthesized ack on a background thread; caller joins it."""
        if not self._acks:
            return None

        pcm = random.choice(self._acks)

        def _speak():
            try:
                tts.speak_pcm(pcm)
            except Exception as e:
                print(f"[Vision] ack playback error: {e}")

        th = threading.Thread(target=_speak, daemon=True)
        th.start()
        return th

    def scene_snapshot(self) -> SceneResult:
        """Grab one frame, detect, serialize. Never raises."""
        t0 = time.monotonic()
        timings: Dict[str, float] = {}
        try:
            frame = self.camera.grab()
            timings["grab"] = time.monotonic() - t0
            if frame is None:
                return SceneResult(ok=False, error="camera returned no frame", timings=timings)

            dets = self.detector.detect(frame.color)
            timings.update(self.detector.last_timings)

            t1 = time.monotonic()
            h, w = frame.color.shape[:2]
            text = serialize(dets, w, h, self.max_objects)
            timings["serialize"] = time.monotonic() - t1
            timings["total"] = time.monotonic() - t0
            return SceneResult(ok=True, text=text, n_detections=len(dets), timings=timings, frame=frame)
        except Exception as e:
            timings["total"] = time.monotonic() - t0
            return SceneResult(ok=False, error=f"{type(e).__name__}: {e}", timings=timings)

    def build_prompt(self, scene_text: str, question: str) -> str:
        return f"{scene_text}\n{self.grounding}\nQuestion: {question}"
