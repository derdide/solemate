"""
visualizer.py
Renders an annotated overlay on a ski photo showing:
  - Detected existing holes (white)
  - Template holes coloured by status: green=reusable, orange=new, red=conflict
  - Ski centreline (cyan) and mounting point crosshair (gold)
  - 100mm scale bar
  - Legend and summary text

Returns JPEG bytes (no disk I/O).
"""
from __future__ import annotations

import io
import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.binding_matcher import BindingConflictResult, HoleStatus, ScaleCalibration


# ---------------------------------------------------------------------------
# Colour palette  (R, G, B)
# ---------------------------------------------------------------------------

_COL = {
    "existing":    (255, 255, 255),
    "reusable":    ( 50, 205,  50),
    "new_hole":    (255, 165,   0),
    "conflict":    (220,  20,  60),
    "centerline":  (  0, 220, 220),
    "mount_point": (255, 215,   0),
    "scale_bar":   (255, 235,   0),
    "legend_bg":   (  0,   0,   0),
    "text_main":   (255, 255, 255),
    "text_dim":    (180, 180, 180),
    "status_ok":   ( 50, 205,  50),
    "status_bad":  (220,  20,  60),
}

_ALPHA_FILL  = 255   # fully opaque fill
_ALPHA_LINE  = 255   # (kept for other uses; hole outlines now black)


def _rgba(color: tuple, alpha: int = 255) -> tuple:
    return (*color, alpha)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_overlay(
    image_bytes: bytes,
    existing_holes_px: list[dict],       # dicts with x_px, y_px, radius_px (optional)
    conflict_result: BindingConflictResult,
    calibration: ScaleCalibration,
    *,
    show_scale_bar: bool = True,
    jpeg_quality: int = 88,
) -> bytes:
    """
    Produce an annotated JPEG image.

    Parameters
    ----------
    image_bytes : raw bytes of the original photo
    existing_holes_px : detected holes in pixel coordinates
    conflict_result : result from binding_matcher.check_binding_conflicts()
    calibration : ScaleCalibration with mm_per_pixel, centerline y, mounting point x
    show_scale_bar : draw a 100mm reference bar
    jpeg_quality : JPEG output quality (1–95)

    Returns
    -------
    JPEG bytes
    """
    # Load image
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    import cv2  # local import to keep module importable without cv2 for tests
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = pil_img.size
    mpp = calibration.mm_per_pixel
    cx = calibration.mounting_point_x_px
    cy = calibration.ski_centerline_y_px
    # Ski axis unit vector (heel→tip in image space)
    adx = getattr(calibration, "axis_dx", -1.0)
    ady = getattr(calibration, "axis_dy",  0.0)
    alen = math.sqrt(adx * adx + ady * ady)
    if alen > 1e-6:
        adx /= alen
        ady /= alen

    # Scale graphical elements relative to image size so they're visible at any resolution.
    # img_scale = 1.0 at 2000px wide; 2.0 at 4000px wide; 0.5 at 1000px wide.
    img_scale = max(0.5, img_w / 2000.0)

    def _mm_to_px(template_x: float, template_y: float) -> tuple[int, int]:
        """
        Convert ski-frame mm coords to image pixel coords.
        Inverse of pixels_to_ski_mm:
            x_px = cx + adx * ski_x / mpp + ady * ski_y / mpp
            y_px = cy + ady * ski_x / mpp - adx * ski_y / mpp
        """
        x_px = cx + adx * template_x / mpp + ady * template_y / mpp
        y_px = cy + ady * template_x / mpp - adx * template_y / mpp
        return int(x_px), int(y_px)

    # ---- Ski centreline (drawn along the ski axis through mounting point) ----
    t_max = float(max(img_w, img_h))
    cl_width = max(2, int(2 * img_scale))
    draw.line(
        [
            (int(cx - adx * t_max), int(cy - ady * t_max)),
            (int(cx + adx * t_max), int(cy + ady * t_max)),
        ],
        fill=_rgba(_COL["centerline"], 170), width=cl_width,
    )

    # ---- Mounting point crosshair ----
    arm = max(24, int(30 * img_scale))
    cross_w = max(3, int(3 * img_scale))
    mpx, mpy = int(cx), int(cy)
    draw.line([(mpx - arm, mpy), (mpx + arm, mpy)], fill=_rgba(_COL["mount_point"], 240), width=cross_w)
    draw.line([(mpx, mpy - arm), (mpx, mpy + arm)], fill=_rgba(_COL["mount_point"], 240), width=cross_w)

    # ---- Existing holes (white fill + bright outline) ----
    # Use a minimum radius of 4.1mm (drill diameter) in pixels, at least 8px for visibility.
    default_r = max(8, int(4.1 / mpp)) if mpp > 0 else 10
    ex_lw = max(2, int(2.5 * img_scale))
    for hole in existing_holes_px:
        x, y = int(hole.get("x_px", 0)), int(hole.get("y_px", 0))
        r = max(int(hole.get("radius_px", default_r)), 8)
        # Solid white fill with black outline — maximum contrast on any background
        draw.ellipse(
            [(x - r, y - r), (x + r, y + r)],
            fill=_rgba(_COL["existing"], 255),
            outline=(0, 0, 0, 255),
            width=ex_lw,
        )

    # ---- Template holes coloured by status ----
    # Radius = half of 4.1mm drill diameter (minimum 8px so it's always visible)
    hole_r = max(8, int(4.1 / (2 * mpp))) if mpp > 0 else 10
    hole_lw = max(2, int(2.5 * img_scale))

    for analysis in conflict_result.holes_analyzed:
        th = analysis.template_hole
        # Use x_abs/y_abs (ski-frame coords) not template_x/y — matters for boot_tip bindings
        px, py = _mm_to_px(th["x_abs"], th["y_abs"])

        if analysis.status == HoleStatus.REUSABLE:
            col = _COL["reusable"]
        elif analysis.status == HoleStatus.CONFLICT:
            col = _COL["conflict"]
        else:
            col = _COL["new_hole"]

        # Solid colored fill with black outline — fully opaque, visible on any background
        draw.ellipse(
            [(px - hole_r, py - hole_r), (px + hole_r, py + hole_r)],
            fill=_rgba(col, _ALPHA_FILL),
            outline=(0, 0, 0, 255),
            width=hole_lw,
        )

        # X through conflict holes — black for contrast against the red fill
        if analysis.status == HoleStatus.CONFLICT:
            x_lw = max(2, int(3 * img_scale))
            draw.line([(px - hole_r, py - hole_r), (px + hole_r, py + hole_r)],
                      fill=(0, 0, 0, 255), width=x_lw)
            draw.line([(px + hole_r, py - hole_r), (px - hole_r, py + hole_r)],
                      fill=(0, 0, 0, 255), width=x_lw)

    # ---- 100mm scale bar ----
    if show_scale_bar and mpp > 0:
        bar_len_px = int(100.0 / mpp)
        bar_h = max(8, int(8 * img_scale))
        bx, by = 20, img_h - int(60 * img_scale)
        draw.rectangle([(bx, by), (bx + bar_len_px, by + bar_h)],
                        fill=_rgba(_COL["scale_bar"], 220))
        _text(draw, (bx, by + bar_h + 4), "100mm", _COL["scale_bar"], size=max(14, int(14 * img_scale)))

    # ---- Legend ----
    _draw_legend(draw, conflict_result, img_w, img_h, img_scale)

    # Composite
    result = Image.alpha_composite(pil_img, overlay).convert("RGB")

    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def _draw_legend(
    draw: ImageDraw.ImageDraw,
    result: BindingConflictResult,
    img_w: int,
    img_h: int,
    img_scale: float = 1.0,
) -> None:
    fs   = max(14, int(14 * img_scale))   # font size
    lx   = max(10, int(10 * img_scale))
    ly   = max(10, int(10 * img_scale))
    lh   = int(fs * 1.6)                  # line height
    sw   = int(fs)                         # colour swatch size
    pad  = max(8, int(8 * img_scale))
    line_count = 6
    legend_w = max(260, int(260 * img_scale))

    # Semi-transparent background
    draw.rectangle(
        [(lx - pad, ly - pad), (lx + legend_w, ly + line_count * lh + pad)],
        fill=_rgba(_COL["legend_bg"], 185),
    )

    # Binding name & BSL
    name = result.binding_name
    if result.variant_id:
        name += f" [{result.variant_id}]"
    if not result.verified:
        name += " ⚠"
    _text(draw, (lx, ly), name, _COL["text_main"], size=fs)
    _text(draw, (lx, ly + lh), f"BSL: {result.bsl_mm:.0f}mm", _COL["text_dim"], size=fs)

    items = [
        (_COL["existing"],  "Existing holes"),
        (_COL["reusable"],  f"Reusable: {result.n_reusable}"),
        (_COL["new_hole"],  f"New holes: {result.n_new_holes}"),
        (_COL["conflict"],  f"Conflicts: {result.n_conflicts}"),
    ]
    for i, (col, label) in enumerate(items):
        y = ly + (i + 2) * lh
        draw.rectangle([(lx, y + int(fs * 0.2)), (lx + sw, y + int(fs * 0.2) + sw)],
                       fill=_rgba(col, 235))
        _text(draw, (lx + sw + pad, y), label, _COL["text_main"], size=fs)

    # Status text at bottom of image
    status_col = _COL["status_ok"] if result.is_mountable else _COL["status_bad"]
    status_text = "MOUNTABLE" if result.is_mountable else "CONFLICT — CANNOT MOUNT"
    bot_margin = int(lh * 1.5)
    _text(draw, (lx, img_h - bot_margin),       f"Status: {status_text}", status_col,  size=fs)
    _text(draw, (lx, img_h - bot_margin - lh),  result.summary[:120],     _COL["text_dim"], size=fs)

    if result.heel_offset_mm != 0.0:
        _text(draw, (lx, img_h - bot_margin - 2 * lh),
              f"Heel offset: {result.heel_offset_mm:+.1f}mm", _COL["text_dim"], size=fs)


def _text(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    color: tuple,
    alpha: int = 230,
    size: int = 14,
) -> None:
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    draw.text(pos, text, fill=(*color, alpha), font=font)
