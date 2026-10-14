"""Scene serializer: detections -> compact text for the LLM.

Left/right are from Oli's own point of view (image left = Oli's left), since Oli
speaks in the first person. No distances: depth is not used in Phase 1.
"""

from typing import List, Tuple

from .detector import Detection

HEADER = "What your camera sees right now:"

_POSITIONS = ["far left", "left", "center", "right", "far right"]
_SMALL_MAX = 0.04    # bbox area / frame area
_MEDIUM_MAX = 0.20

_IRREGULAR_PLURALS = {
    "person": "people",
    "knife": "knives",
    "sheep": "sheep",
    "scissors": "scissors",
    "skis": "skis",
}


def _position(cx_frac: float) -> str:
    return _POSITIONS[max(0, min(4, int(cx_frac * 5)))]


def _size(area_frac: float) -> str:
    if area_frac < _SMALL_MAX:
        return "small"
    if area_frac < _MEDIUM_MAX:
        return "medium"
    return "large"


def _plural(label: str) -> str:
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    return label + "s"


def serialize(dets: List[Detection], frame_w: int, frame_h: int, max_objects: int = 10) -> str:
    """Group detections by label; list count, size (single objects) and position(s)."""
    if not dets:
        return f"{HEADER} nothing recognizable."

    groups = {}  # label -> list of (cx_frac, area_frac, score); dict keeps first-seen order
    for d in sorted(dets, key=lambda d: -d.score):
        cx = ((d.x1 + d.x2) / 2.0) / frame_w
        area = max(0.0, (d.x2 - d.x1) * (d.y2 - d.y1)) / float(frame_w * frame_h)
        groups.setdefault(d.label, []).append((cx, area, d.score))

    lines = [HEADER]
    for label, items in list(groups.items())[:max_objects]:
        items.sort(key=lambda it: it[0])  # left -> right
        n = len(items)
        if n == 1:
            cx, area, _ = items[0]
            lines.append(f"- 1 {label} ({_size(area)}, {_position(cx)})")
        else:
            positions: List[str] = []
            for cx, _, _ in items:
                p = _position(cx)
                if p not in positions:
                    positions.append(p)
            lines.append(f"- {n} {_plural(label)} ({', '.join(positions)})")
    return "\n".join(lines)
