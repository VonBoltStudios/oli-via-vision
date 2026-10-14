#!/usr/bin/env python3
"""Standalone vision test: grab a frame, detect, print the scene text and per-stage timing.

    python3 vision_test.py                    # 3 runs with config.yaml
    python3 vision_test.py -n 10 --save vision_frames
    python3 vision_test.py --gate "what's on the table?"   # test the gate only (no camera)

Swap models by editing vision.detector in config.yaml (or pass another config file).
"""

import argparse
import os
import time

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("-n", type=int, default=3, help="number of grab+detect runs")
    ap.add_argument("--save", metavar="DIR", help="save each captured frame as PNG in DIR")
    ap.add_argument("--gate", metavar="TEXT", help="only check whether TEXT would trigger vision")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    vcfg = cfg.get("vision") or {}

    if args.gate is not None:
        from modules.vision.gate import VisionGate
        match = VisionGate(vcfg.get("patterns")).match(args.gate)
        print(f"{args.gate!r} -> {'MATCH ' + repr(match) if match else 'no match'}")
        return 0

    from modules.vision import Vision

    vision = Vision(vcfg)
    vision.warmup()

    if args.save:
        from PIL import Image
        os.makedirs(args.save, exist_ok=True)

    for i in range(1, args.n + 1):
        print(f"\n--- run {i}/{args.n} ---")
        scene = vision.scene_snapshot()
        if not scene.ok:
            print(f"FAILED: {scene.error}  ({scene.timing_line()})")
            continue
        print(scene.text)
        print(f"{scene.n_detections} detection(s) | {scene.timing_line()}")
        if args.save and scene.frame is not None:
            path = os.path.join(args.save, f"frame_{time.strftime('%H%M%S')}_{i}.png")
            Image.fromarray(scene.frame.color).save(path)
            print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
