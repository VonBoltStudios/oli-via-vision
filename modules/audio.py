"""Audio capture: VAD-gated recording via LimX SDK WebSocket stream."""

from __future__ import annotations

import collections
import json
import threading
import time
import uuid

import numpy as np
import websocket


class WSAudioCapture:
    """
    Drop-in replacement for AudioCapture that sources audio from the LimX SDK
    WebSocket stream (notify_audio_capture) instead of sounddevice.

    The WS connection is opened on construction and streaming starts immediately.
    Call close() when done (or use as a context manager).
    """

    def __init__(
        self,
        host: str,
        port: int,
        sample_rate: int,
        channels: int,
        chunk_ms: int,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_frames = int(sample_rate * chunk_ms / 1000)

        self._host = host
        self._port = port
        self._accid: str | None = None
        self._accid_event = threading.Event()
        self._pending: dict = {}
        self._pending_lock = threading.Lock()

        # Thread-safe PCM buffer (int16 samples)
        self._buf: collections.deque[np.ndarray] = collections.deque()
        self._buf_lock = threading.Lock()
        self._buf_event = threading.Event()  # set whenever new data arrives

        self._ws: websocket.WebSocketApp | None = None
        self._connected = threading.Event()

        self._connect()
        self._start_capture()

    # ── WS infrastructure ─────────────────────────────────────────────────────

    def _guid(self) -> str:
        return str(uuid.uuid4())

    def _send(self, title: str, data: dict | None = None, timeout: float = 10.0):
        guid = self._guid()
        msg = {
            "accid": self._accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": guid,
            "data": data or {},
        }
        evt = threading.Event()
        holder: dict = {"resp": None}
        with self._pending_lock:
            self._pending[guid] = (evt, holder)
        self._ws.send(json.dumps(msg, separators=(",", ":")))
        if not evt.wait(timeout):
            with self._pending_lock:
                self._pending.pop(guid, None)
            raise TimeoutError(f"'{title}' timed out after {timeout}s")
        with self._pending_lock:
            self._pending.pop(guid, None)
        return holder["resp"] or {}

    def _on_message(self, ws, raw: str):
        root = json.loads(raw)
        if root.get("accid") and not self._accid:
            self._accid = root["accid"]
            self._accid_event.set()

        title = root.get("title", "")

        if title == "notify_audio_capture":
            d = root.get("data", {})
            samples = d.get("samples")
            if samples:
                chunk = np.array(samples, dtype=np.int16)
                with self._buf_lock:
                    self._buf.append(chunk)
                self._buf_event.set()

        elif title.startswith("response_"):
            guid = root.get("guid", "")
            with self._pending_lock:
                entry = self._pending.get(guid)
            if entry:
                evt, holder = entry
                holder["resp"] = root.get("data", {})
                evt.set()

    def _connect(self):
        ready = threading.Event()

        def _on_open(ws):
            self._connected.set()
            ready.set()

        self._ws = websocket.WebSocketApp(
            f"ws://{self._host}:{self._port}",
            on_open=_on_open,
            on_message=self._on_message,
            on_close=lambda ws, c, m: None,
        )
        t = threading.Thread(target=self._ws.run_forever, daemon=True)
        t.start()

        if not ready.wait(10):
            raise ConnectionError(f"WS connect timeout ({self._host}:{self._port})")
        if not self._accid_event.wait(10):
            raise ConnectionError("ACCID not received from SDK")
        print(f"[WSAudioCapture] Connected. ACCID: {self._accid}")

    def _start_capture(self):
        r = self._send("request_audio_capture", {"streaming": 1})
        print(f"[WSAudioCapture] Streaming started: {r.get('result', '?')}")

    def close(self):
        try:
            self._send("request_audio_capture", {"streaming": 0}, timeout=5)
        except Exception:
            pass
        if self._ws:
            self._ws.close()

    def flush(self):
        """Discard buffered audio — call after TTS playback to prevent self-response."""
        with self._buf_lock:
            self._buf.clear()
            self._buf_event.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _drain_buf(self) -> np.ndarray | None:
        """Pop all queued chunks and concatenate. Returns None if empty."""
        with self._buf_lock:
            if not self._buf:
                return None
            chunks = list(self._buf)
            self._buf.clear()
            self._buf_event.clear()
        return np.concatenate(chunks)

    def _rms_float(self, samples_i16: np.ndarray) -> float:
        """Return RMS amplitude normalised to 0-1 (float32)."""
        f = samples_i16.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(f ** 2)))

    def _read_chunk_blocking(self, timeout: float) -> np.ndarray | None:
        """
        Block until at least chunk_frames samples are in the buffer, or timeout.
        Returns chunk_frames of int16 samples, or None on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._buf_lock:
                total = sum(len(c) for c in self._buf)
            if total >= self.chunk_frames:
                # drain exactly chunk_frames
                with self._buf_lock:
                    merged = np.concatenate(list(self._buf))
                    self._buf.clear()
                    self._buf_event.clear()
                chunk = merged[: self.chunk_frames]
                remainder = merged[self.chunk_frames :]
                if len(remainder):
                    with self._buf_lock:
                        self._buf.appendleft(remainder)
                return chunk
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._buf_event.wait(timeout=min(remaining, 0.02))
            self._buf_event.clear()

    # ── Public interface (mirrors AudioCapture) ───────────────────────────────

    def wait_and_record(
        self,
        wait_timeout_s: float,
        silence_threshold_ms: int,
        max_record_s: int,
        amplitude_threshold: float = 0.01,
    ) -> np.ndarray | None:
        """
        Phase 1: wait up to wait_timeout_s for speech onset.
        Phase 2: record until silence_threshold_ms of silence, capped at max_record_s.
        Returns int16 audio, or None if no speech detected.
        """
        chunk_frames = self.chunk_frames
        silence_frames_needed = int(
            self.sample_rate * silence_threshold_ms / 1000 / chunk_frames
        )
        max_record_chunks = int(self.sample_rate * max_record_s / chunk_frames)
        wait_deadline = time.monotonic() + wait_timeout_s

        frames: list[np.ndarray] = []
        silent_chunks = 0

        # Phase 1: wait for speech onset
        while time.monotonic() < wait_deadline:
            chunk = self._read_chunk_blocking(timeout=wait_deadline - time.monotonic())
            if chunk is None:
                return None
            if self._rms_float(chunk) >= amplitude_threshold:
                frames.append(chunk)
                break
        else:
            return None

        # Phase 2: record until silence
        for _ in range(max_record_chunks):
            chunk = self._read_chunk_blocking(timeout=2.0)
            if chunk is None:
                break
            frames.append(chunk)
            if self._rms_float(chunk) < amplitude_threshold:
                silent_chunks += 1
                if silent_chunks >= silence_frames_needed:
                    break
            else:
                silent_chunks = 0

        if not frames:
            return None
        return np.concatenate(frames)

    def record_until_silence(
        self,
        silence_threshold_ms: int,
        max_record_s: int,
        amplitude_threshold: float = 0.01,
    ) -> np.ndarray:
        """
        Record immediately (no onset wait). Returns full clip as int16 array.
        """
        chunk_frames = self.chunk_frames
        silence_frames_needed = int(
            self.sample_rate * silence_threshold_ms / 1000 / chunk_frames
        )
        max_chunks = int(self.sample_rate * max_record_s / chunk_frames)

        frames: list[np.ndarray] = []
        silent_chunks = 0

        print("  [listening...]")
        for _ in range(max_chunks):
            chunk = self._read_chunk_blocking(timeout=2.0)
            if chunk is None:
                break
            frames.append(chunk)
            if self._rms_float(chunk) < amplitude_threshold:
                silent_chunks += 1
                if silent_chunks >= silence_frames_needed:
                    break
            else:
                silent_chunks = 0

        if not frames:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(frames)
