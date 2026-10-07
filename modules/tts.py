"""Text-to-speech via Piper TTS → LimX SDK WebSocket PCM streaming."""

import json
import threading
import time
import uuid
import wave

import numpy as np
import onnxruntime as ort
import websocket
from piper.voice import PiperVoice
from piper.config import SynthesisConfig


class TTS:
    def __init__(
        self,
        voice: str,
        speed: float,
        sdk_host: str = "10.192.1.2",
        sdk_port: int = 5000,
        chunk_ms: int = 100,
        buffer_ms: int = 1000,
        lead_in_ms: int = 0,
        **kwargs,
    ):
        self.lead_in_ms = lead_in_ms
        self.chunk_ms = chunk_ms
        self.buffer_ms = buffer_ms
        self.syn_config = SynthesisConfig(length_scale=1.0 / speed)

        print(f"[TTS] Loading Piper voice: {voice}")
        self.model = PiperVoice.load(voice)
        self.model.session = ort.InferenceSession(
            voice, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.sample_rate = self.model.config.sample_rate
        print(f"[TTS] Ready. Sample rate: {self.sample_rate}Hz")

        self._host = sdk_host
        self._port = sdk_port
        self._accid: str | None = None
        self._accid_event = threading.Event()
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None

        self._connect()

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
        if title.startswith("response_"):
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
            ready.set()

        self._ws = websocket.WebSocketApp(
            f"ws://{self._host}:{self._port}",
            on_open=_on_open,
            on_message=self._on_message,
            on_close=lambda ws, c, m: None,
        )
        self._ws.sock_opt = [
            ("socket", "SO_SNDBUF", 8 * 1024 * 1024),
            ("socket", "SO_RCVBUF", 8 * 1024 * 1024),
        ]
        t = threading.Thread(target=self._ws.run_forever, daemon=True)
        t.start()

        if not ready.wait(10):
            raise ConnectionError(f"TTS WS connect timeout ({self._host}:{self._port})")
        if not self._accid_event.wait(10):
            raise ConnectionError("TTS ACCID not received from SDK")
        print(f"[TTS] WS connected. ACCID: {self._accid}")

    def close(self):
        if self._ws:
            self._ws.close()

    # ── Speak ─────────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Synthesize text with Piper and stream PCM to SDK speaker."""
        if not text:
            return

        chunks = list(self.model.synthesize(text, syn_config=self.syn_config))
        if not chunks:
            return

        audio_f32 = np.concatenate([c.audio_float_array for c in chunks])
        pcm = (audio_f32 * 32767).clip(-32767, 32767).astype(np.int16)

        # Leading silence: a cold SDK speaker can clip the start of short clips ("Ok.", "Ready.")
        if self.lead_in_ms > 0:
            pcm = np.concatenate([np.zeros(int(self.sample_rate * self.lead_in_ms / 1000), dtype=np.int16), pcm])

        sr = self.sample_rate
        ch = 1
        chunk_samples = int(sr * self.chunk_ms / 1000) * ch
        buffer_s = self.buffer_ms / 1000.0
        total = len(pcm)

        self._send("request_audio_playback_control", {"enable": 1})

        offset = 0
        start = time.monotonic()
        sent_s = 0.0

        try:
            while offset < total:
                played_s = time.monotonic() - start
                buffered = sent_s - played_s
                if buffered >= buffer_s:
                    time.sleep(min(buffered - buffer_s, 0.02))
                    continue

                end = min(offset + chunk_samples, total)
                chunk = pcm[offset:end]
                chunk_dur = (end - offset) / (sr * ch)

                self._send("request_audio_play_data", {
                    "sample_rate": sr,
                    "channels": ch,
                    "samples": chunk.tolist(),
                }, timeout=5)

                offset = end
                sent_s += chunk_dur

        except Exception as e:
            print(f"[TTS] Playback error: {e}")

        remaining = sent_s - (time.monotonic() - start)
        time.sleep(max(0.3, remaining + 0.2))
        self._send("request_audio_playback_control", {"enable": 0})

    # ── Pre-recorded audio ────────────────────────────────────────────────────

    @staticmethod
    def _wav_duration(path: str) -> float | None:
        """Duration in seconds of a local WAV, or None if unreadable (URL, other host)."""
        try:
            with wave.open(path, "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            return None

    def play_file(self, path: str, timeout: float = 180.0) -> bool:
        """Play a pre-recorded WAV (path/URL on the robot) via request_audio_play_file.

        Blocks until playback should be finished: if the SDK holds its response until
        playback ends we return then; otherwise we sleep out the WAV's remaining duration.
        """
        start = time.monotonic()
        try:
            resp = self._send("request_audio_play_file", {"file_path": path}, timeout=timeout)
        except Exception as e:
            print(f"[TTS] play_file error: {e}")
            return False
        if resp.get("result") != "success":
            print(f"[TTS] play_file failed: {resp.get('message', resp.get('result'))}")
            return False
        dur = self._wav_duration(path)
        if dur:
            remaining = dur - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining + 0.2)
        return True
