"""
main.py — FastAPI web application for ski binding hole conflict detection.

Endpoints:
  GET  /                  → Serve index.html
  GET  /static/*          → Serve static assets
  GET  /api/health        → Health check
  GET  /api/bindings      → List all bindings
  GET  /api/template/{id} → Hole positions for a binding+BSL
  POST /api/analyze       → Full analysis pipeline
  POST /api/visualize     → Re-render overlay for a specific binding result
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np

from src.binding_matcher import (
    BindingConflictResult,
    ScaleCalibration,
    check_binding_conflicts,
    compute_absolute_holes,
    load_binding_db,
    pixels_to_ski_mm,
    rank_all_bindings,
)
from src.hole_detector import DetectedHole
from src.visualizer import draw_overlay

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "templates" / "binding_db.json"

app = FastAPI(title="Ski Binding Conflict Detector", version="1.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load binding database once at startup
_binding_db: dict = {}


@app.on_event("startup")
def _load_db():
    global _binding_db
    _binding_db = load_binding_db(DB_PATH)
    n = len(_binding_db.get("bindings", []))
    print(f"[startup] Loaded {n} bindings from {DB_PATH}")


# ---------------------------------------------------------------------------
# Static / root routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
def serve_root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "bindings_loaded": len(_binding_db.get("bindings", [])),
    }


# ---------------------------------------------------------------------------
# Bindings list
# ---------------------------------------------------------------------------

@app.get("/api/bindings")
def list_bindings(category: str = "all"):
    bindings = _binding_db.get("bindings", [])
    if category != "all":
        bindings = [b for b in bindings if b.get("category") == category]
    return [
        {
            "id": b["id"],
            "name": b["name"],
            "aliases": b.get("aliases", []),
            "category": b.get("category"),
            "verified": b.get("verified", True),
            "bsl_range_mm": b.get("bsl_range_mm"),
            "mounting_reference": b.get("mounting_reference"),
            "notes": b.get("notes", ""),
            "has_variants": bool(b.get("variants")),
        }
        for b in bindings
    ]


# ---------------------------------------------------------------------------
# Template hole positions
# ---------------------------------------------------------------------------

@app.get("/api/template/{binding_id}")
def get_template(binding_id: str, bsl: float = 300.0, variant: Optional[str] = None):
    binding = next(
        (b for b in _binding_db.get("bindings", []) if b["id"] == binding_id), None
    )
    if binding is None:
        raise HTTPException(status_code=404, detail=f"Binding '{binding_id}' not found")

    try:
        holes = compute_absolute_holes(binding, bsl, mounting_point_x=0.0, variant_id=variant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "binding_id": binding_id,
        "binding_name": binding["name"],
        "bsl_mm": bsl,
        "variant_id": variant,
        "holes": [
            {"x": h["template_x"], "y": h["template_y"], "label": h["label"], "unit": h["unit"]}
            for h in holes
        ],
    }


# ---------------------------------------------------------------------------
# Calibration endpoint — perspective correction + scale + axis
# ---------------------------------------------------------------------------

@app.post("/api/calibrate")
async def api_calibrate(
    image: UploadFile = File(...),
    tape_points_json: str = Form(...),     # JSON [{x_px, y_px}, …]
    tape_spacing_mm: float = Form(100.0),  # real-world spacing between consecutive marks
    mounting_point_x_px: float = Form(...),
    mounting_point_y_px: float = Form(...),
    axis_dx: float = Form(-1.0),
    axis_dy: float = Form(0.0),
):
    """
    Compute scale + optional 1-D perspective correction from tape marks.
    Returns a rectified JPEG (or original if < 3 marks) and calibration params.
    """
    import traceback
    try:
        image_bytes = await image.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")

        raw_pts = json.loads(tape_points_json)
        tape_points = [(float(p["x_px"]), float(p["y_px"])) for p in raw_pts]
        n = len(tape_points)
        if n < 2:
            raise HTTPException(400, "Need at least 2 tape marks")

        mpx, mpy = float(mounting_point_x_px), float(mounting_point_y_px)
        adx, ady = float(axis_dx), float(axis_dy)

        # Scale from mean tape interval
        u_obs = sorted(
            (x - mpx) * adx + (y - mpy) * ady for x, y in tape_points
        )
        spans_px = [abs(u_obs[i + 1] - u_obs[i]) for i in range(n - 1)]
        mean_span_px = float(np.mean(spans_px))
        mm_per_pixel = tape_spacing_mm / mean_span_px

        # Perspective correction when 3+ marks available
        perspective_corrected = False
        if n >= 3:
            from src.calibrator import rectify_image
            img_out, (mpx, mpy) = rectify_image(img, tape_points, (mpx, mpy), adx, ady)
            perspective_corrected = True
        else:
            img_out = img

        _, buf = cv2.imencode(".jpg", img_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "rectified_image_base64": img_b64,
            "calibration": {
                "mm_per_pixel": mm_per_pixel,
                "mounting_point_x_px": mpx,
                "ski_centerline_y_px": mpy,
                "axis_dx": adx,
                "axis_dy": ady,
                "perspective_corrected": perspective_corrected,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    bsl_mm: float = Form(...),
    category: str = Form("all"),
    min_separation_mm: float = Form(14.0),
    top_n: int = Form(20),
    # Multi-BSL probing: also test at bsl_mm ± bsl_test_range in bsl_test_step increments.
    # Set bsl_test_step=0 to test only at the exact bsl_mm.
    bsl_test_step: int = Form(10),
    bsl_test_range: int = Form(30),
    # Manual calibration overrides (optional)
    mm_per_pixel: Optional[float] = Form(None),
    mounting_point_x_px: Optional[float] = Form(None),
    mounting_point_y_px: Optional[float] = Form(None),
    # Ski axis direction in image space (unit vector from heel to tip).
    # axis_dx=-1, axis_dy=0  →  horizontal ski, tip to the LEFT  (default)
    # axis_dx=0,  axis_dy=-1 →  vertical ski, tip UPWARD
    axis_dx: Optional[float] = Form(None),
    axis_dy: Optional[float] = Form(None),
    # If provided, use these holes directly and skip hole detection entirely
    override_holes_json: Optional[str] = Form(None),
):
    """
    Full analysis pipeline:
      1. Read uploaded image
      2. Convert manually provided holes + calibration to mm
      3. Rank all compatible bindings
      4. Render overlay for best result
      5. Return JSON with results + base64 image
    """
    import traceback
    try:
     return await _analyze_inner(
        image, bsl_mm, category, min_separation_mm, top_n,
        bsl_test_step, bsl_test_range,
        mm_per_pixel, mounting_point_x_px, mounting_point_y_px,
        axis_dx, axis_dy,
        override_holes_json,
     )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


async def _analyze_inner(
    image, bsl_mm, category, min_separation_mm, top_n,
    bsl_test_step, bsl_test_range,
    mm_per_pixel, mounting_point_x_px, mounting_point_y_px,
    axis_dx, axis_dy,
    override_holes_json,
):
    image_bytes = await image.read()

    # Build calibration from manually provided values
    calibration: Optional[ScaleCalibration] = None
    if mm_per_pixel is not None and mounting_point_x_px is not None and mounting_point_y_px is not None:
        calibration = ScaleCalibration(
            mm_per_pixel=mm_per_pixel,
            ski_centerline_y_px=mounting_point_y_px,
            mounting_point_x_px=mounting_point_x_px,
            axis_dx=axis_dx if axis_dx is not None else -1.0,
            axis_dy=axis_dy if axis_dy is not None else 0.0,
        )

    # Parse manually provided holes (empty list = fresh ski, still valid)
    holes_px: list[DetectedHole] = []
    if override_holes_json:
        try:
            raw = json.loads(override_holes_json)
            holes_px = [
                DetectedHole(
                    x_px=float(h["x_px"]), y_px=float(h["y_px"]),
                    radius_px=float(h.get("radius_px", 5)),
                    confidence=1.0, source="manual",
                )
                for h in raw
            ]
        except Exception as e:
            print(f"override_holes_json parse error: {e}")

    calibration_available = calibration is not None
    mounting_point_known = (
        calibration is not None
        and calibration.mounting_point_x_px != 0.0
    )

    # Convert holes to mm coords
    existing_holes_mm: list[dict] = []
    if mounting_point_known and holes_px:
        existing_holes_mm = pixels_to_ski_mm(holes_px, calibration)

    # Rank bindings — runs whenever mounting point is set (empty holes = fresh ski)
    results: list = []
    if mounting_point_known:
        results = rank_all_bindings(
            _binding_db,
            bsl_mm=bsl_mm,
            existing_holes_mm=existing_holes_mm,
            mounting_point_x=0.0,
            min_separation_mm=min_separation_mm,
            category_filter=category,
            bsl_test_step=bsl_test_step,
            bsl_test_range=bsl_test_range,
        )
        results = results[:top_n]

    # Render overlay for best result
    output_image_b64: Optional[str] = None
    if mounting_point_known and results:
        best = results[0]
        try:
            img_bytes = draw_overlay(
                image_bytes=image_bytes,
                existing_holes_px=[
                    {"x_px": h.x_px, "y_px": h.y_px, "radius_px": h.radius_px}
                    for h in holes_px
                ],
                conflict_result=best,
                calibration=calibration,
            )
            output_image_b64 = base64.b64encode(img_bytes).decode()
        except Exception as e:
            print(f"Visualizer error: {e}")

    holes_px_out = [
        {"x_px": round(h.x_px), "y_px": round(h.y_px),
         "radius_px": round(h.radius_px, 1), "confidence": round(h.confidence, 2),
         "source": h.source}
        for h in holes_px
    ]

    return {
        "detected_holes": len(holes_px),
        "holes_px": holes_px_out,
        "calibration_auto": False,
        "calibration_available": calibration_available,
        "mounting_point_known": mounting_point_known,
        "calibration": {
            "mm_per_pixel": calibration.mm_per_pixel if calibration else None,
            "mounting_point_x_px": calibration.mounting_point_x_px if calibration else None,
            "ski_centerline_y_px": calibration.ski_centerline_y_px if calibration else None,
            "axis_dx": getattr(calibration, "axis_dx", -1.0) if calibration else -1.0,
            "axis_dy": getattr(calibration, "axis_dy",  0.0) if calibration else  0.0,
        },
        "results": [_result_to_dict(r) for r in results],
        "identify_matches": [],   # placeholder — populated by dedicated button in UI
        "output_image_base64": output_image_b64,
    }


# ---------------------------------------------------------------------------
# Re-visualize endpoint (click a table row → new overlay)
# ---------------------------------------------------------------------------

@app.post("/api/visualize")
async def visualize(
    image: UploadFile = File(...),
    binding_id: str = Form(...),
    bsl_mm: float = Form(...),
    variant_id: Optional[str] = Form(None),
    min_separation_mm: float = Form(14.0),
    existing_holes_json: str = Form("[]"),      # JSON array of {x_abs, y_abs}
    existing_holes_px_json: str = Form("[]"),   # JSON array of {x_px, y_px, radius_px}
    mm_per_pixel: float = Form(...),
    mounting_point_x_px: float = Form(...),
    mounting_point_y_px: float = Form(...),
    axis_dx: float = Form(-1.0),
    axis_dy: float = Form(0.0),
    heel_offset_mm: Optional[float] = Form(None),   # override automatic heel search
    mount_offset_mm: float = Form(0.0),             # shift mounting point along ski axis
):
    """Re-render the overlay for a specific binding (interactive overlay controls)."""
    import traceback
    try:
        return await _visualize_inner(
            image, binding_id, bsl_mm, variant_id, min_separation_mm,
            existing_holes_json, existing_holes_px_json,
            mm_per_pixel, mounting_point_x_px, mounting_point_y_px,
            axis_dx, axis_dy, heel_offset_mm, mount_offset_mm,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


async def _visualize_inner(
    image, binding_id, bsl_mm, variant_id, min_separation_mm,
    existing_holes_json, existing_holes_px_json,
    mm_per_pixel, mounting_point_x_px, mounting_point_y_px,
    axis_dx, axis_dy, heel_offset_mm, mount_offset_mm,
):
    image_bytes = await image.read()

    binding = next(
        (b for b in _binding_db.get("bindings", []) if b["id"] == binding_id), None
    )
    if binding is None:
        raise HTTPException(status_code=404, detail=f"Binding '{binding_id}' not found")

    try:
        existing_holes = json.loads(existing_holes_json)
    except Exception:
        existing_holes = []

    try:
        existing_holes_px = json.loads(existing_holes_px_json)
    except Exception:
        existing_holes_px = []

    # Apply mount offset: shift mounting point in pixel space and adjust
    # existing hole mm coords so they remain correct relative to the new origin.
    if mount_offset_mm != 0.0:
        offset_px = mount_offset_mm / mm_per_pixel
        mounting_point_x_px = mounting_point_x_px + axis_dx * offset_px
        mounting_point_y_px = mounting_point_y_px + axis_dy * offset_px
        existing_holes = [
            {**h, "x_abs": h.get("x_abs", 0) - mount_offset_mm} for h in existing_holes
        ]

    calibration = ScaleCalibration(
        mm_per_pixel=mm_per_pixel,
        ski_centerline_y_px=mounting_point_y_px,
        mounting_point_x_px=mounting_point_x_px,
        axis_dx=axis_dx,
        axis_dy=axis_dy,
    )

    result = check_binding_conflicts(
        binding, bsl_mm, existing_holes,
        mounting_point_x=0.0,
        min_separation_mm=min_separation_mm,
        variant_id=variant_id,
        heel_offset_mm=heel_offset_mm,
    )

    img_bytes = draw_overlay(
        image_bytes=image_bytes,
        existing_holes_px=existing_holes_px,
        conflict_result=result,
        calibration=calibration,
    )

    return {
        "result": _result_to_dict(result),
        "output_image_base64": base64.b64encode(img_bytes).decode(),
    }


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _result_to_dict(r: BindingConflictResult) -> dict:
    binding = next(
        (b for b in _binding_db.get("bindings", []) if b["id"] == r.binding_id), {}
    )
    adj_range = binding.get("heel_unit", {}).get("adjustment_range_mm", 0.0) if binding else 0.0
    bsl_range = binding.get("bsl_range_mm")
    return {
        "binding_id": r.binding_id,
        "binding_name": r.binding_name,
        "bsl_mm": r.bsl_mm,
        "bsl_range_mm": bsl_range,
        "adjustment_range_mm": adj_range,
        "variant_id": r.variant_id,
        "verified": r.verified,
        "is_mountable": r.is_mountable,
        "n_new_holes": r.n_new_holes,
        "n_reusable": r.n_reusable,
        "n_conflicts": r.n_conflicts,
        "conflict_score": r.conflict_score,
        "heel_offset_mm": r.heel_offset_mm,
        "summary": r.summary,
        "holes": [
            {
                "label": a.template_hole["label"],
                "unit": a.template_hole["unit"],
                "template_x": a.template_hole["template_x"],
                "template_y": a.template_hole["template_y"],
                "status": a.status.value,
                "distance_mm": round(a.distance_mm, 2) if a.distance_mm is not None else None,
            }
            for a in r.holes_analyzed
        ],
    }
