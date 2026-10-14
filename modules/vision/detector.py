"""Object detectors behind one small interface.

Phase 1 ships the RF-DETR Nano model from Akshay's reference project
(oli-yolo-reference) with two backends:

  rfdetr_trt   TensorRT FP16 engine (fast path on Oli)
  rfdetr_onnx  ONNX Runtime, CUDA with CPU fallback (works without TensorRT)

Pre/postprocessing is adapted from the reference project's onnx_detector.py and
shared by both backends. To try another model, add a Detector subclass here and
register it in build_detector(); nothing else changes.

RF-DETR specifics (verified in the reference project):
  input   (1, 3, 384, 384) float32 NCHW, ImageNet-normalized, whole frame squashed
  output  'dets'   (1, 300, 4)  normalized cxcywh
  output  'labels' (1, 300, 91) raw logits, per-class sigmoid; 91 slots = sparse COCO
          category IDs (index == category ID), no background slot
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .coco_classes import class_name

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# apt-installed TensorRT lives outside the venv; these are appended (never prepended)
# to sys.path so the venv's own packages (e.g. NumPy) still win.
_SYSTEM_DIST_DIRS = ["/usr/lib/python3.10/dist-packages", "/usr/lib/python3/dist-packages"]


@dataclass
class Detection:
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector:
    last_timings: Dict[str, float]

    def load(self) -> None:
        raise NotImplementedError

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """rgb: (H, W, 3) uint8, RGB order. Sets self.last_timings (seconds)."""
        raise NotImplementedError


class _RFDETRBase(Detector):
    def __init__(self, model_path: str, confidence: float = 0.5):
        self.model_path = os.path.expanduser(model_path)
        self.confidence = confidence
        self.input_width = 384
        self.input_height = 384
        self.num_slots = 91
        self.last_timings = {}

    # -- subclasses provide: load() and _infer(tensor) -> (boxes (300,4), logits (300,slots))

    def _infer(self, tensor: np.ndarray):
        raise NotImplementedError

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(rgb).resize((self.input_width, self.input_height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)

    def postprocess(self, boxes: np.ndarray, logits: np.ndarray, width: int, height: int) -> List[Detection]:
        slots = np.arange(logits.shape[-1])
        if logits.shape[-1] != 91:
            # custom checkpoint: last slot is background
            logits = logits[:, :-1]
            slots = slots[:-1]

        scores_all = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))  # per-class sigmoid
        scores = scores_all.max(axis=-1)
        class_ids = slots[scores_all.argmax(axis=-1)]

        keep = scores > self.confidence
        if not np.any(keep):
            return []

        cx, cy, bw, bh = boxes[keep].T
        out = []
        for cid, s, cxi, cyi, bwi, bhi in zip(class_ids[keep], scores[keep], cx, cy, bw, bh):
            label = class_name(int(cid))
            if label == "N/A":
                continue
            out.append(Detection(
                label=label,
                score=float(s),
                x1=float(max(0.0, (cxi - bwi / 2.0) * width)),
                y1=float(max(0.0, (cyi - bhi / 2.0) * height)),
                x2=float(min(width, (cxi + bwi / 2.0) * width)),
                y2=float(min(height, (cyi + bhi / 2.0) * height)),
            ))
        return out

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        h, w = rgb.shape[:2]
        t0 = time.monotonic()
        tensor = self.preprocess(rgb)
        t1 = time.monotonic()
        boxes, logits = self._infer(tensor)
        t2 = time.monotonic()
        dets = self.postprocess(boxes, logits, w, h)
        t3 = time.monotonic()
        self.last_timings = {"preprocess": t1 - t0, "infer": t2 - t1, "post": t3 - t2}
        return dets


class RFDETRONNX(_RFDETRBase):
    def load(self) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            self.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        _, _, self.input_height, self.input_width = inputs[0].shape
        self.input_name = inputs[0].name
        names = [o.name for o in self.session.get_outputs()]
        # match outputs by name, never by shape (see reference project notes)
        self._dets_idx = next((i for i, n in enumerate(names) if "dets" in n.lower()), None)
        self._labels_idx = next((i for i, n in enumerate(names) if "labels" in n.lower()), None)
        if self._dets_idx is None or self._labels_idx is None:
            raise ValueError(f"Could not find 'dets' and 'labels' outputs in {names}")
        self.num_slots = self.session.get_outputs()[self._labels_idx].shape[-1]
        print(f"[Vision] RF-DETR ONNX loaded: {self.model_path} "
              f"({self.input_width}x{self.input_height}, {self.num_slots} slots, "
              f"providers={self.session.get_providers()})")

    def _infer(self, tensor):
        outputs = self.session.run(None, {self.input_name: tensor})
        return outputs[self._dets_idx][0], outputs[self._labels_idx][0]


def _import_tensorrt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        for d in _SYSTEM_DIST_DIRS:
            if os.path.isdir(os.path.join(d, "tensorrt")) and d not in sys.path:
                sys.path.append(d)
        import tensorrt as trt
        return trt


def _import_cudart():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart  # older cuda-python layout
    return cudart


class RFDETRTensorRT(_RFDETRBase):
    def load(self) -> None:
        trt = _import_tensorrt()
        cudart = _import_cudart()
        self._cudart = cudart

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f:
            engine_bytes = f.read()
        self._runtime = trt.Runtime(logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(
                f"Could not deserialize {self.model_path} (TensorRT {trt.__version__}). Engines are "
                f"not portable: rebuild on this Oli with trtexec from the .onnx (about 2 minutes).")
        self.context = self._engine.create_execution_context()

        err, self._stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")

        input_names, output_names, output_shapes = [], [], {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
                output_shapes[name] = self._engine.get_tensor_shape(name)
        if len(input_names) != 1:
            raise ValueError(f"Expected 1 input tensor, engine has {input_names}")

        self.input_name = input_names[0]
        input_shape = self._engine.get_tensor_shape(self.input_name)
        _, _, self.input_height, self.input_width = input_shape

        dets_name = next((n for n in output_names if "dets" in n.lower()), None)
        labels_name = next((n for n in output_names if "labels" in n.lower()), None)
        if dets_name is None or labels_name is None:
            raise ValueError(f"Could not find 'dets' and 'labels' outputs in {output_names}")
        self._dets_idx = output_names.index(dets_name)
        self._labels_idx = output_names.index(labels_name)
        self.num_slots = output_shapes[labels_name][-1]

        itemsize = np.dtype(np.float32).itemsize
        err, self._d_input = cudart.cudaMalloc(int(np.prod(input_shape)) * itemsize)
        self._check(err, "cudaMalloc(input)")
        self.context.set_tensor_address(self.input_name, int(self._d_input))

        self._d_outputs, self._h_outputs = [], []
        for name in output_names:
            shape = output_shapes[name]
            err, d_ptr = cudart.cudaMalloc(int(np.prod(shape)) * itemsize)
            self._check(err, f"cudaMalloc({name})")
            self._d_outputs.append(d_ptr)
            self._h_outputs.append(np.empty(shape, dtype=np.float32))
            self.context.set_tensor_address(name, int(d_ptr))

        print(f"[Vision] RF-DETR TensorRT loaded: {self.model_path} "
              f"({self.input_width}x{self.input_height}, {self.num_slots} slots, TensorRT {trt.__version__})")

    def _check(self, err, what: str) -> None:
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{what} failed: {err}")

    def _infer(self, tensor):
        cudart = self._cudart
        kind = cudart.cudaMemcpyKind
        err, = cudart.cudaMemcpyAsync(
            self._d_input, tensor.ctypes.data, tensor.nbytes, kind.cudaMemcpyHostToDevice, self._stream)
        self._check(err, "H2D copy")
        self.context.execute_async_v3(stream_handle=self._stream)
        for d_ptr, h_buf in zip(self._d_outputs, self._h_outputs):
            err, = cudart.cudaMemcpyAsync(
                h_buf.ctypes.data, d_ptr, h_buf.nbytes, kind.cudaMemcpyDeviceToHost, self._stream)
            self._check(err, "D2H copy")
        err, = cudart.cudaStreamSynchronize(self._stream)
        self._check(err, "stream sync")
        return self._h_outputs[self._dets_idx][0], self._h_outputs[self._labels_idx][0]

    def __del__(self):
        try:
            cudart = self._cudart
            cudart.cudaFree(self._d_input)
            for d_ptr in self._d_outputs:
                cudart.cudaFree(d_ptr)
            cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass


_BACKENDS = {
    "rfdetr_trt": RFDETRTensorRT,
    "rfdetr_onnx": RFDETRONNX,
}


def build_detector(dcfg: dict) -> Detector:
    backend = dcfg.get("backend", "rfdetr_trt")
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown vision.detector.backend {backend!r}; choose from {sorted(_BACKENDS)}")
    return _BACKENDS[backend](
        model_path=dcfg["model_path"],
        confidence=float(dcfg.get("confidence", 0.5)),
    )
