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
import os
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.binding_matcher import (
    BindingConflictResult,
    ScaleCalibration,
    check_binding_conflicts,
    compute_absolute_holes,
    identify_previous_binding,
    load_binding_db,
    pixels_to_ski_mm,
    rank_all_bindings,
)
from src.hole_detector import DetectedHole, HoleDetector
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
# Authentication
# ---------------------------------------------------------------------------

APP_TOKEN = os.environ.get("APP_TOKEN", "")


def _verify_token(token: Optional[str] = None) -> None:
    """Dependency: verify Bearer token from query param or Authorization header."""
    # If no APP_TOKEN is configured, skip auth (dev mode)
    if not APP_TOKEN:
        return
    if token != APP_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")


# FastAPI dependency — token can come from ?token= query parameter
def _auth(token: Optional[str] = None):
    _verify_token(token)


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
        "auth_enabled": bool(APP_TOKEN),
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
# Main analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(
    _auth: Annotated[None, Depends(_auth)],
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
      2. Detect holes + calibration (OpenCV → Claude API fallback)
      3. Convert pixel coords to mm
      4. Rank all compatible bindings
      5. Identify previously mounted binding
      6. Render overlay for best result
      7. Return JSON with results + base64 image
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

    # Build manual calibration if user supplied values
    manual_cal = None
    if mm_per_pixel is not None and mounting_point_x_px is not None and mounting_point_y_px is not None:
        manual_cal = ScaleCalibration(
            mm_per_pixel=mm_per_pixel,
            ski_centerline_y_px=mounting_point_y_px,
            mounting_point_x_px=mounting_point_x_px,
            axis_dx=axis_dx if axis_dx is not None else -1.0,
            axis_dy=axis_dy if axis_dy is not None else 0.0,
        )

    # If user edited holes manually and provided calibration, skip detection entirely
    if override_holes_json and manual_cal:
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
            holes_px = []
        calibration = manual_cal
    else:
        # Normal detection (OpenCV + Claude fallback)
        detector = HoleDetector()
        holes_px, calibration = detector.detect(image_bytes, manual_calibration=manual_cal)

    calibration_auto = manual_cal is None
    calibration_available = calibration is not None

    # Convert to mm coords (requires calibration + mounting point set)
    existing_holes_mm: list[dict] = []
    mounting_point_known = (
        calibration is not None
        and calibration.mounting_point_x_px != 0.0
    )
    if mounting_point_known:
        existing_holes_mm = pixels_to_ski_mm(holes_px, calibration)

    # Rank bindings (only meaningful if we have mm coords)
    results: list = []
    if existing_holes_mm:
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

    # Identify previously mounted binding — BSL-independent scan over all sizes
    # Works from mm coords when available, reports nothing if calibration missing
    identify_matches = []
    if existing_holes_mm:
        identify_matches = identify_previous_binding(existing_holes_mm, _binding_db)

    # Render overlay for best result (if calibration + mounting point available)
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

    # Always return detected holes in pixel space so the UI can show them
    # and let the user click to set the mounting point / tape reference
    holes_px_out = [
        {"x_px": round(h.x_px), "y_px": round(h.y_px),
         "radius_px": round(h.radius_px, 1), "confidence": round(h.confidence, 2),
         "source": h.source}
        for h in holes_px
    ]

    return {
        "detected_holes": len(holes_px),
        "holes_px": holes_px_out,
        "calibration_auto": calibration_auto,
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
        "identify_matches": identify_matches,
        "output_image_base64": output_image_b64,
    }


# ---------------------------------------------------------------------------
# Re-visualize endpoint (click a table row → new overlay)
# ---------------------------------------------------------------------------

@app.post("/api/visualize")
async def visualize(
    _auth: Annotated[None, Depends(_auth)],
    image: UploadFile = File(...),
    binding_id: str = Form(...),
    bsl_mm: float = Form(...),
    variant_id: Optional[str] = Form(None),
    min_separation_mm: float = Form(14.0),
    existing_holes_json: str = Form("[]"),   # JSON array of {x_abs, y_abs}
    mm_per_pixel: float = Form(...),
    mounting_point_x_px: float = Form(...),
    mounting_point_y_px: float = Form(...),
    axis_dx: float = Form(-1.0),
    axis_dy: float = Form(0.0),
):
    """Re-render the overlay for a specific binding (user clicked a table row)."""
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
    )

    # We need pixel-space hole list for the overlay
    # Since we don't re-detect, send empty (existing holes rendered from mm coords directly)
    img_bytes = draw_overlay(
        image_bytes=image_bytes,
        existing_holes_px=[],   # not available without re-detecting
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
    return {
        "binding_id": r.binding_id,
        "binding_name": r.binding_name,
        "bsl_mm": r.bsl_mm,
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
