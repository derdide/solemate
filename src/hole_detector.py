"""
hole_detector.py
DetectedHole dataclass — geometry result of manual hole marking.
Auto-detection has been removed; holes are always entered manually via the UI.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectedHole:
    x_px: float
    y_px: float
    radius_px: float
    confidence: float   # 0.0–1.0
    source: str         # "manual"
