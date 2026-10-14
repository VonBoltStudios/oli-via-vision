"""Head-camera frame grabber (Intel RealSense via pyrealsense2).

The pipeline is opened per grab: start (~0.05 s), discard a few warm-up frames so
auto-exposure settles, keep the last one, stop. On Oli this fits inside the spoken
acknowledgement, and it avoids holding the USB camera open between questions.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    color: np.ndarray            # (H, W, 3) uint8, RGB
    depth: Optional[np.ndarray]  # (H, W) uint16 depth units, or None (depth disabled)
    t: float                     # time.monotonic() at capture


class HeadCamera:
    def __init__(
        self,
        serial: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_frames: int = 10,
        timeout_ms: int = 3000,
        depth: bool = False,
        depth_width: int = 848,
        depth_height: int = 480,
    ):
        import pyrealsense2 as rs  # imported lazily so the rest of VIA runs without it

        self._rs = rs
        self.serial = str(serial)
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = max(1, warmup_frames)
        self.timeout_ms = timeout_ms
        self.depth = depth
        self.depth_width = depth_width
        self.depth_height = depth_height

    def grab(self) -> Optional[Frame]:
        """Capture one frame from the configured camera; None on any failure."""
        rs = self._rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.depth:
            cfg.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)

        pipe = rs.pipeline()
        try:
            pipe.start(cfg)
            frames = None
            for _ in range(self.warmup_frames):
                frames = pipe.wait_for_frames(self.timeout_ms)
            color = frames.get_color_frame()
            if not color:
                return None
            color_arr = np.asanyarray(color.get_data()).copy()
            depth_arr = None
            if self.depth:
                d = frames.get_depth_frame()
                if d:
                    depth_arr = np.asanyarray(d.get_data()).copy()
            return Frame(color=color_arr, depth=depth_arr, t=time.monotonic())
        except Exception as e:
            print(f"[Vision] camera error ({self.serial}): {e}")
            return None
        finally:
            try:
                pipe.stop()
            except Exception:
                pass
