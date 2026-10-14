#!/usr/bin/env python3
"""Verify camera targeting: list RealSense devices and save one color frame from each.

Look at the saved images to confirm which serial is the head camera, then make sure
vision.camera.serial in config.yaml matches it.

    python3 vision_probe.py [config.yaml] [--out probe]
"""

import argparse
import os
import time

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--out", default="probe", help="directory for saved frames")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ccfg = (cfg.get("vision") or {}).get("camera") or {}
    configured = str(ccfg.get("serial", ""))

    import pyrealsense2 as rs
    from PIL import Image

    from modules.vision.camera import HeadCamera

    devices = list(rs.context().query_devices())
    if not devices:
        print("No RealSense devices found.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    print(f"Found {len(devices)} device(s). Configured head serial: {configured or '(none)'}\n")

    found_configured = False
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        tag = "  <-- configured head camera" if serial == configured else ""
        found_configured |= serial == configured
        print(f"{name}  serial {serial}  firmware {fw}{tag}")

        cam = HeadCamera(
            serial=serial,
            width=ccfg.get("width", 1280),
            height=ccfg.get("height", 720),
            fps=ccfg.get("fps", 30),
            warmup_frames=ccfg.get("warmup_frames", 10),
            timeout_ms=ccfg.get("timeout_ms", 3000),
        )
        t0 = time.monotonic()
        frame = cam.grab()
        if frame is None:
            print("  grab FAILED (see error above; is another process holding this camera?)\n")
            continue
        path = os.path.join(args.out, f"{serial}_color.png")
        Image.fromarray(frame.color).save(path)
        print(f"  grab ok in {time.monotonic() - t0:.2f}s -> {path}\n")

    if configured and not found_configured:
        print(f"WARNING: configured serial {configured} was not found among the devices above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
