"""Audio capture: VAD-gated recording."""

import numpy as np
import sounddevice as sd


class AudioCapture:
    def __init__(self, sample_rate: int, channels: int, chunk_ms: int):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_frames = int(sample_rate * chunk_ms / 1000)

    def wait_and_record(
        self,
        wait_timeout_s: float,
        silence_threshold_ms: int,
        max_record_s: int,
        amplitude_threshold: float = 0.01,
    ) -> np.ndarray | None:
        """
        Phase 1: wait up to wait_timeout_s for speech onset (amplitude above threshold).
        Phase 2: once speech starts, record until silence_threshold_ms of silence, capped at max_record_s.
        Returns int16 audio, or None if no speech detected within wait_timeout_s.
        """
        chunk_frames = self.chunk_frames
        wait_chunks = int(self.sample_rate * wait_timeout_s / chunk_frames)
        silence_frames_needed = int(
            self.sample_rate * silence_threshold_ms / 1000 / chunk_frames
        )
        max_record_chunks = int(self.sample_rate * max_record_s / chunk_frames)

        frames: list[np.ndarray] = []
        silent_chunks = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=chunk_frames,
        ) as stream:
            # Phase 1: wait for speech onset
            for _ in range(wait_chunks):
                chunk, _ = stream.read(chunk_frames)
                chunk = chunk.flatten()
                if np.abs(chunk).mean() >= amplitude_threshold:
                    frames.append(chunk)
                    break
            else:
                return None  # timed out — no speech detected

            # Phase 2: record until silence
            for _ in range(max_record_chunks):
                chunk, _ = stream.read(chunk_frames)
                chunk = chunk.flatten()
                frames.append(chunk)

                if np.abs(chunk).mean() < amplitude_threshold:
                    silent_chunks += 1
                    if silent_chunks >= silence_frames_needed:
                        break
                else:
                    silent_chunks = 0

        audio = np.concatenate(frames)
        return (audio * 32768).astype(np.int16)

    def record_until_silence(
        self,
        silence_threshold_ms: int,
        max_record_s: int,
        amplitude_threshold: float = 0.01,
    ) -> np.ndarray:
        """
        Record immediately (no speech-onset wait). Returns full clip as int16 array.
        Kept for push-to-talk mode where the user has already indicated readiness.
        """
        chunk_frames = self.chunk_frames
        silence_frames_needed = int(
            self.sample_rate * silence_threshold_ms / 1000 / chunk_frames
        )
        max_chunks = int(self.sample_rate * max_record_s / chunk_frames)

        frames: list[np.ndarray] = []
        silent_chunks = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=chunk_frames,
        ) as stream:
            print("  [listening...]")
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_frames)
                chunk = chunk.flatten()
                frames.append(chunk)

                if np.abs(chunk).mean() < amplitude_threshold:
                    silent_chunks += 1
                    if silent_chunks >= silence_frames_needed:
                        break
                else:
                    silent_chunks = 0

        audio = np.concatenate(frames)
        return (audio * 32768).astype(np.int16)
