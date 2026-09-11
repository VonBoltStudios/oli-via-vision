"""Wake word detection via openWakeWord."""

import numpy as np
from openwakeword.model import Model


class WakeWordDetector:
    def __init__(self, phrase: str, threshold: float):
        self.phrase = phrase
        self.threshold = threshold
        # openWakeWord uses pre-trained models; load the one matching the phrase
        self.model = Model(wakeword_models=[phrase], inference_framework="onnx")

    def check(self, audio_chunk: np.ndarray) -> bool:
        """Process one audio chunk. Returns True if wake word detected."""
        # openWakeWord expects int16 or float32 at 16kHz
        prediction = self.model.predict(audio_chunk)
        score = prediction.get(self.phrase, 0.0)
        return score >= self.threshold
