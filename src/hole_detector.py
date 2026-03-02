"""
hole_detector.py
Hybrid hole detection pipeline:
  1. OpenCV (local, no API cost) — tries HoughCircles + dark blob detection
  2. Ollama Vision fallback — if OLLAMA_URL is set (local, free, no API key)
  3. Claude Vision API fallback — if ANTHROPIC_API_KEY is set and Ollama not used/failed
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedHole:
    x_px: float
    y_px: float
    radius_px: float
    confidence: float   # 0.0–1.0
    source: str         # "opencv" or "claude_api"


@dataclass
class ScaleCalibration:
    mm_per_pixel: float
    ski_centerline_y_px: float   # y pixel of mounting point (= ski centerline)
    mounting_point_x_px: float   # x pixel of mounting point
    # Unit vector from heel toward tip in IMAGE pixel space.
    # Default (-1, 0) = ski is horizontal with tip to the LEFT.
    # For vertical ski with tip UP: (-0, -1) = (0, -1).
    axis_dx: float = -1.0
    axis_dy: float = 0.0
    reference_points: list = field(default_factory=list)
    reference_length_mm: float = 0.0


# ---------------------------------------------------------------------------
# HoleDetector
# ---------------------------------------------------------------------------

class HoleDetector:
    MAX_DIM_PX = 4000

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-6"):
        # --- Claude ---
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model or os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")
        self._use_claude = os.environ.get("USE_CLAUDE_FALLBACK", "true").lower() == "true"
        self._client = None  # Lazy init

        # --- Ollama ---
        self._ollama_url  = os.environ.get("OLLAMA_URL", "").rstrip("/")   # e.g. http://localhost:11434
        self._ollama_model = os.environ.get("OLLAMA_MODEL", "llava")
        self._use_ollama  = bool(self._ollama_url)

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def detect(
        self,
        image_bytes: bytes,
        manual_calibration: Optional[ScaleCalibration] = None,
    ) -> tuple[list[DetectedHole], Optional[ScaleCalibration]]:
        """
        Detect holes and derive scale calibration from raw image bytes.
        If manual_calibration is provided it takes precedence over auto-detection.

        Returns (holes, calibration). calibration may be None if auto-detect fails
        AND no manual override was given.

        Strategy:
          1. Run OpenCV on a resized copy (fast, free)
          2. If OpenCV calibration failed OR no holes found → call Claude Vision API
             (Claude is much better at reading measuring tapes in real photos)
          3. Merge hole lists; prefer Claude calibration when OpenCV calibration failed
        """
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image (unsupported format?)")

        # Resize for OpenCV processing (keep original bytes for Claude)
        img_resized, scale_factor = self._resize(img)

        holes_cv, cal_cv = self._detect_opencv(img_resized, scale_factor)

        # Scale CV results back to original pixel coordinates immediately
        if scale_factor != 1.0:
            for h in holes_cv:
                h.x_px /= scale_factor
                h.y_px /= scale_factor
                h.radius_px /= scale_factor
            if cal_cv:
                cal_cv.ski_centerline_y_px /= scale_factor
                cal_cv.mounting_point_x_px /= scale_factor
                cal_cv.mm_per_pixel *= scale_factor

        # Decide whether to call an AI Vision fallback:
        # - Call if calibration not found AND no manual calibration was given
        # - Also call if no holes found at all (detection needed regardless)
        # - SKIP if manual calibration provided (saves time + API cost on re-analysis)
        cal_missing = cal_cv is None and manual_calibration is None
        no_holes = len(holes_cv) == 0
        needs_ai = cal_missing or no_holes

        holes_ai: list[DetectedHole] = []
        cal_ai: Optional[ScaleCalibration] = None

        if needs_ai and self._use_ollama:
            print(f"[detector] Calling Ollama ({self._ollama_model}) (cal_missing={cal_missing}, no_holes={no_holes})")
            holes_ai, cal_ai = self._detect_ollama(image_bytes)
            if not holes_ai and not cal_ai:
                print("[detector] Ollama returned nothing — trying Claude next")

        if needs_ai and not holes_ai and not cal_ai and self._use_claude and self._api_key:
            print(f"[detector] Calling Claude Vision API (cal_missing={cal_missing}, no_holes={no_holes})")
            holes_ai, cal_ai = self._detect_claude(image_bytes)

        # Merge holes (deduplicate)
        holes = _merge_holes(holes_cv, holes_ai)

        # Calibration priority: manual > AI (better at tape reading) > OpenCV
        if manual_calibration is not None:
            calibration = manual_calibration
        elif cal_missing and cal_ai is not None:
            calibration = cal_ai
        elif cal_cv is not None:
            calibration = cal_ai if cal_ai is not None else cal_cv
        else:
            calibration = cal_ai  # may still be None

        ai_src = "ollama" if (holes_ai and self._use_ollama) else "claude" if holes_ai else "none"
        print(f"[detector] holes={len(holes)} (cv={len(holes_cv)}, ai/{ai_src}={len(holes_ai)}), "
              f"calibration={'OK' if calibration else 'MISSING'}")

        return holes, calibration

    # -----------------------------------------------------------------------
    # Resize helper
    # -----------------------------------------------------------------------

    def _resize(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.MAX_DIM_PX:
            return img, 1.0
        factor = self.MAX_DIM_PX / max_dim
        new_w, new_h = int(w * factor), int(h * factor)
        return cv2.resize(img, (new_w, new_h)), factor

    # -----------------------------------------------------------------------
    # OpenCV pipeline
    # -----------------------------------------------------------------------

    def _detect_opencv(
        self, img: np.ndarray, scale_factor: float
    ) -> tuple[list[DetectedHole], Optional[ScaleCalibration]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        calibration = self._calibrate_scale(img, gray, scale_factor)
        centerline_y = self._detect_ski_centerline(gray)
        if centerline_y is not None and calibration is not None:
            calibration.ski_centerline_y_px = centerline_y

        holes = self._detect_circles(gray, calibration)
        return holes, calibration

    def _calibrate_scale(
        self, img: np.ndarray, gray: np.ndarray, scale_factor: float
    ) -> Optional[ScaleCalibration]:
        """
        Detect measuring tape scale using frequency analysis.
        Looks for a strip with regular tick marks and estimates mm/pixel.

        Returns None if the tape cannot be reliably detected.
        """
        h, w = gray.shape

        # Scan horizontal strips for periodic high-contrast pattern
        strip_height = max(5, h // 40)
        best_period = None
        best_confidence = 0.0
        best_y = h // 2

        for y_start in range(0, h - strip_height, strip_height * 2):
            strip = gray[y_start:y_start + strip_height, :]
            profile = strip.mean(axis=0).astype(float)

            # Normalize
            profile -= profile.mean()
            if profile.std() < 5:
                continue

            # FFT to find dominant period
            fft = np.abs(np.fft.rfft(profile))
            freqs = np.fft.rfftfreq(len(profile))

            # Ignore DC and very long periods (>w/2 px) and very short (<5px)
            min_freq = 1.0 / (w / 2)
            max_freq = 1.0 / 5
            mask = (freqs > min_freq) & (freqs < max_freq)
            if not mask.any():
                continue

            fft_masked = fft.copy()
            fft_masked[~mask] = 0
            peak_freq = freqs[np.argmax(fft_masked)]
            period_px = 1.0 / peak_freq if peak_freq > 0 else 0
            confidence = fft_masked.max() / (fft.sum() + 1e-6)

            # Measuring tape: tick marks at 1mm, major at 10mm, 5mm
            # Plausible period range: 3–50 pixels (for typical photo scales)
            if 3 <= period_px <= 50 and confidence > best_confidence:
                best_confidence = confidence
                best_period = period_px
                best_y = y_start + strip_height // 2

        if best_period is None or best_confidence < 0.05:
            return None

        # Assume the detected period corresponds to 1mm ticks on the tape
        mm_per_pixel = 1.0 / best_period

        # Estimate mounting point as horizontal centre (user can override)
        mounting_point_x = gray.shape[1] / 2.0

        return ScaleCalibration(
            mm_per_pixel=mm_per_pixel,
            ski_centerline_y_px=gray.shape[0] / 2.0,  # updated by centerline detector
            mounting_point_x_px=mounting_point_x,
            reference_points=[],
            reference_length_mm=0.0,
        )

    def _detect_ski_centerline(self, gray: np.ndarray) -> Optional[float]:
        """
        Find y-coordinate of ski centreline.
        Strategy: detect long horizontal edges and take the average of top/bottom ski edges.
        """
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, math.pi / 180,
            threshold=80,
            minLineLength=gray.shape[1] // 4,
            maxLineGap=50,
        )
        if lines is None:
            return None

        # Keep only roughly horizontal lines
        horizontal = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            if abs(y2 - y1) < gray.shape[0] // 10:
                horizontal.append((y1 + y2) / 2)

        if len(horizontal) < 2:
            return None

        horizontal.sort()
        # Take the two extreme y values (top and bottom ski edge)
        return (horizontal[0] + horizontal[-1]) / 2.0

    def _detect_circles(
        self, gray: np.ndarray, calibration: Optional[ScaleCalibration]
    ) -> list[DetectedHole]:
        """
        Two-pass circle detection:
          Pass 1: HoughCircles on blurred grayscale
          Pass 2: Dark blob (Otsu threshold) + circularity filter
        Results are merged and deduplicated.
        """
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Estimate plausible hole radius range in pixels
        if calibration and calibration.mm_per_pixel > 0:
            mpp = calibration.mm_per_pixel
            min_r = max(3, int(1.5 / mpp))
            max_r = max(min_r + 2, int(7.0 / mpp))
        else:
            # Heuristic: hole radius ≈ 0.15–0.4% of image width
            min_r = max(3, w // 700)
            max_r = max(min_r + 5, w // 250)

        holes: list[DetectedHole] = []

        # --- Pass 1: HoughCircles ---
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_r * 3,
            param1=60,
            param2=20,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for x, y, r in circles[0]:
                holes.append(DetectedHole(
                    x_px=float(x), y_px=float(y),
                    radius_px=float(r), confidence=0.7, source="opencv",
                ))

        # --- Pass 2: Dark blob detection ---
        _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < math.pi * min_r ** 2 * 0.5:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < 0.65:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (min_r <= radius <= max_r):
                continue
            # Deduplicate against existing Hough detections
            if not any(
                math.hypot(h.x_px - cx, h.y_px - cy) < radius
                for h in holes
            ):
                holes.append(DetectedHole(
                    x_px=float(cx), y_px=float(cy),
                    radius_px=float(radius), confidence=0.55, source="opencv",
                ))

        return holes

    # -----------------------------------------------------------------------
    # Ollama Vision fallback (local, no API key required)
    # -----------------------------------------------------------------------

    def _detect_ollama(
        self, image_bytes: bytes
    ) -> tuple[list[DetectedHole], Optional[ScaleCalibration]]:
        """
        Send image to a local Ollama instance with a vision-capable model
        (e.g. llava, llava-llama3, moondream2, minicpm-v).

        Set env vars:
          OLLAMA_URL   = http://localhost:11434   (or remote host)
          OLLAMA_MODEL = llava                    (any vision model you have pulled)
        """
        import urllib.request
        import urllib.error

        media_type = _guess_media_type(image_bytes)
        image_b64  = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Ollama /api/generate with images list
        payload = json.dumps({
            "model":  self._ollama_model,
            "stream": False,
            "images": [image_b64],
            "prompt": (
                "You are analyzing a photo of a ski to detect binding screw holes and measure scale.\n\n"
                "CRITICAL: Focus EXCLUSIVELY on the ski itself — the long, narrow object in the image. "
                "Completely ignore anything outside the ski surface (floor, carpet, walls, hands, "
                "tape measure body, background objects, shadows, etc.).\n\n"
                "WHAT BINDING SCREW HOLES LOOK LIKE:\n"
                "- Small, dark, circular depressions drilled INTO the ski topsheet surface\n"
                "- Approximately 4mm diameter in real life (appear as tiny dots in most photos)\n"
                "- Always arranged in SYMMETRIC GROUPS: pairs on the left AND right of the ski centreline\n"
                "- Typically 4 holes per binding unit, 8 holes total for a full binding set\n"
                "- Located on the flat top surface of the ski, in the mid-section (not at tip or tail)\n\n"
                "DO NOT REPORT: wood grain, texture, graphics, rivets, scratches, background objects.\n\n"
                "YOUR TASKS:\n"
                "A) Find all drill holes on the ski surface — report only high-confidence circular holes\n"
                "B) Read the measuring tape: find two clearly labelled tick marks a known distance apart\n"
                "C) Find the ski centreline: the longitudinal axis along the middle of the ski width\n"
                "D) Find the mounting point mark if visible\n\n"
                "Return ONLY valid JSON, no extra text:\n"
                '{"holes":[{"x_px":123,"y_px":456,"radius_px":6,"confidence":0.9}],'
                '"scale":{"reference_x1_px":100,"reference_y1_px":500,'
                '"reference_x2_px":400,"reference_y2_px":500,'
                '"reference_length_mm":100,"mm_per_pixel":0.33},'
                '"ski_centerline_y_px":400,"mounting_point_x_px":null,'
                '"ski_orientation":"horizontal_tip_left",'
                '"notes":"brief summary"}'
            ),
        }).encode("utf-8")

        url = f"{self._ollama_url}/api/generate"
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                text = json.loads(resp.read().decode("utf-8")).get("response", "")
        except urllib.error.URLError as e:
            print(f"[ollama] connection error: {e}")
            return [], None
        except Exception as e:
            print(f"[ollama] error: {e}")
            return [], None

        # Parse the JSON response (same structure as Claude prompt)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            print(f"[ollama] no JSON in response: {text[:200]}")
            return [], None
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"[ollama] JSON parse error: {e}")
            return [], None

        holes = [
            DetectedHole(
                x_px=float(h["x_px"]),
                y_px=float(h["y_px"]),
                radius_px=float(h.get("radius_px", 6)),
                confidence=float(h.get("confidence", 0.8)),
                source="ollama",
            )
            for h in (data.get("holes") or [])
            if h.get("x_px") is not None
        ]

        print(f"[ollama] notes: {data.get('notes', '')}")
        print(f"[ollama] orientation: {data.get('ski_orientation', '?')}, holes: {len(holes)}")

        calibration: Optional[ScaleCalibration] = None
        scale = data.get("scale")
        if scale and scale.get("mm_per_pixel"):
            mpp = float(scale["mm_per_pixel"])
            rx1, ry1 = scale.get("reference_x1_px"), scale.get("reference_y1_px")
            rx2, ry2 = scale.get("reference_x2_px"), scale.get("reference_y2_px")
            ref_mm   = scale.get("reference_length_mm", 0)
            if rx1 and rx2 and ref_mm:
                dist_px = math.sqrt((rx2-rx1)**2 + (ry2-ry1)**2)
                if dist_px > 5:
                    mpp = float(ref_mm) / dist_px
                    print(f"[ollama] scale recalculated: {ref_mm}mm / {dist_px:.1f}px = {mpp:.4f} mm/px")
            mp_x = data.get("mounting_point_x_px") or 0.0
            cl_y = data.get("ski_centerline_y_px")
            if cl_y is None:
                cl_y = sum(h.y_px for h in holes) / len(holes) if holes else 500.0
            calibration = ScaleCalibration(
                mm_per_pixel=mpp,
                ski_centerline_y_px=float(cl_y),
                mounting_point_x_px=float(mp_x),
                reference_points=[(rx1 or 0, ry1 or 0), (rx2 or 0, ry2 or 0)],
                reference_length_mm=float(ref_mm),
            )

        return holes, calibration

    # -----------------------------------------------------------------------
    # Claude Vision API fallback
    # -----------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _detect_claude(
        self, image_bytes: bytes
    ) -> tuple[list[DetectedHole], Optional[ScaleCalibration]]:
        """
        Send image to Claude Vision API with a structured JSON prompt.
        Returns (holes, calibration).
        """
        # Determine media type from magic bytes
        media_type = _guess_media_type(image_bytes)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        prompt = (
            "You are analyzing a photo of a ski to detect binding screw holes and measure scale.\n\n"
            "CRITICAL: Focus EXCLUSIVELY on the ski itself — the long, narrow object in the image. "
            "Completely ignore anything outside the ski surface (floor, carpet, walls, hands, "
            "tape measure body, background objects, shadows, etc.).\n\n"
            "WHAT BINDING SCREW HOLES LOOK LIKE:\n"
            "- Small, dark, circular depressions drilled INTO the ski topsheet surface\n"
            "- Approximately 4mm diameter in real life (appear as tiny dots in most photos)\n"
            "- Always arranged in SYMMETRIC GROUPS: pairs on the left AND right of the ski centreline\n"
            "- Typically 4 holes per binding unit, 8 holes total for a full binding set\n"
            "- Located on the flat top surface of the ski, in the mid-section (not at tip or tail)\n"
            "- May have screw remnants, be empty, or slightly clogged — still circular\n\n"
            "DO NOT REPORT:\n"
            "- Wood grain, surface texture, graphics, or decorative patterns\n"
            "- Rivets, ventilation holes, or design elements on the ski\n"
            "- Damage, scratches, discoloration, reflections, or glare spots\n"
            "- Anything in the background or outside the ski boundary\n"
            "- Single isolated spots (binding holes always come in symmetric pairs)\n\n"
            "YOUR TASKS:\n"
            "A) Find all drill holes on the ski surface only — report only high-confidence circular holes\n"
            "B) Read the measuring tape: find two clearly labelled tick marks a known distance apart "
            "(e.g. 10cm and 20cm = 100mm). Pick marks that are clearly visible and far enough apart.\n"
            "C) Find the ski centreline: the longitudinal axis along the middle of the ski width\n"
            "D) Find the mounting point mark if visible (a centre arrow or line on the ski topsheet)\n\n"
            "Return ONLY valid JSON, no extra text:\n"
            "{\n"
            '  "holes": [\n'
            '    {"x_px": 123, "y_px": 456, "radius_px": 6, "confidence": 0.9}\n'
            "  ],\n"
            '  "scale": {\n'
            '    "reference_x1_px": 100, "reference_y1_px": 500,\n'
            '    "reference_x2_px": 400, "reference_y2_px": 500,\n'
            '    "reference_length_mm": 100,\n'
            '    "mm_per_pixel": 0.33\n'
            "  },\n"
            '  "ski_centerline_y_px": 400,\n'
            '  "mounting_point_x_px": null,\n'
            '  "ski_orientation": "horizontal_tip_left",\n'
            '  "notes": "ski colour, approx hole count, tape visibility, any issues"\n'
            "}\n\n"
            "RULES:\n"
            "- Pixel coordinates: top-left = (0, 0)\n"
            "- Only report holes with confidence >= 0.7; omit uncertain detections\n"
            "- Holes must be in symmetric pairs — if you see one side, look for the mirror hole\n"
            "- ski_orientation: 'horizontal_tip_left', 'horizontal_tip_right', or 'vertical_tip_up'\n"
            "- scale: null if tape not clearly visible\n"
            "- mounting_point_x_px: null if no centre mark visible\n"
            "- radius_px: actual hole size in pixels (typically 3–15px depending on photo scale)"
        )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self._model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
        except Exception as e:
            print(f"Claude API error: {e}")
            return [], None

        text = response.content[0].text
        # Extract JSON block
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            return [], None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return [], None

        holes = [
            DetectedHole(
                x_px=float(h["x_px"]),
                y_px=float(h["y_px"]),
                radius_px=float(h.get("radius_px", 6)),
                confidence=float(h.get("confidence", 0.8)),
                source="claude_api",
            )
            for h in data.get("holes") or []
            if h.get("x_px") is not None
        ]

        print(f"[claude] notes: {data.get('notes', '')}")
        print(f"[claude] orientation: {data.get('ski_orientation', 'unknown')}, holes found: {len(holes)}")

        calibration: Optional[ScaleCalibration] = None
        scale = data.get("scale")
        if scale and scale.get("mm_per_pixel"):
            # Recalculate mm_per_pixel from reference points for accuracy
            mpp = float(scale["mm_per_pixel"])
            rx1 = scale.get("reference_x1_px")
            ry1 = scale.get("reference_y1_px")
            rx2 = scale.get("reference_x2_px")
            ry2 = scale.get("reference_y2_px")
            ref_mm = scale.get("reference_length_mm", 0)
            if rx1 and rx2 and ref_mm:
                dist_px = math.sqrt((rx2-rx1)**2 + (ry2-ry1)**2)
                if dist_px > 5:
                    mpp = float(ref_mm) / dist_px
                    print(f"[claude] scale recalculated: {ref_mm}mm / {dist_px:.1f}px = {mpp:.4f} mm/px")

            # Estimate mounting point as image horizontal centre if not explicitly found
            mp_x = data.get("mounting_point_x_px")
            if mp_x is None:
                # Can't know without user input — set to 0 (unknown, user must set)
                mp_x = 0.0

            cl_y = data.get("ski_centerline_y_px")
            if cl_y is None:
                cl_y = sum(h.y_px for h in holes) / len(holes) if holes else 500.0

            calibration = ScaleCalibration(
                mm_per_pixel=mpp,
                ski_centerline_y_px=float(cl_y),
                mounting_point_x_px=float(mp_x),
                reference_points=[
                    (rx1 or 0, ry1 or 0),
                    (rx2 or 0, ry2 or 0),
                ],
                reference_length_mm=float(ref_mm),
            )
            calibration.ski_orientation = data.get("ski_orientation", "horizontal_tip_left")

        return holes, calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_media_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # safe default


def _merge_holes(
    a: list[DetectedHole], b: list[DetectedHole], min_dist_px: float = 10.0
) -> list[DetectedHole]:
    """Merge two hole lists, deduplicating by proximity."""
    merged = list(a)
    for hole in b:
        if not any(
            math.hypot(hole.x_px - m.x_px, hole.y_px - m.y_px) < min_dist_px
            for m in merged
        ):
            merged.append(hole)
    return merged
