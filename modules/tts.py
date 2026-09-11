"""Text-to-speech via Piper TTS (Python API)."""

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
from piper.config import SynthesisConfig


class TTS:
    def __init__(self, voice: str, speed: float, output_device=None, **kwargs):
        """
        voice: path to the .onnx model file
        speed: speech rate multiplier (1.0 = normal)
        """
        self.output_device = output_device
        self.syn_config = SynthesisConfig(length_scale=1.0 / speed)
        print(f"[TTS] Loading Piper voice: {voice}")
        self.model = PiperVoice.load(voice)
        self.sample_rate = self.model.config.sample_rate
        print(f"[TTS] Ready. Sample rate: {self.sample_rate}Hz")

    def speak(self, text: str):
        """Synthesize text and play via speaker."""
        if not text:
            return

        chunks = list(self.model.synthesize(text, syn_config=self.syn_config))
        if not chunks:
            return

        audio = np.concatenate([c.audio_float_array for c in chunks])
        sd.play(audio, self.sample_rate, device=self.output_device)
        sd.wait()
