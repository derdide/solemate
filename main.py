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
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders

import cv2
import numpy as np
from PIL import Image

from src.binding_matcher import (
    BindingConflictResult,
    ScaleCalibration,
    check_binding_conflicts,
    compute_absolute_holes,
    load_binding_db,
    pixels_to_ski_mm,
    rank_all_bindings,
)
from src.calibrator import rectify_image
from src.hole_detector import DetectedHole
from src.visualizer import draw_overlay

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "templates" / "binding_db.json"

log = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Limits — every endpoint is reachable without authentication, so nothing a
# client sends may decide how much memory, CPU or disk a request costs.
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024)
MAX_IMAGE_PIXELS = int(float(os.environ.get("MAX_IMAGE_MEGAPIXELS", "50")) * 1_000_000)
MAX_FORM_OVERHEAD_BYTES = 2 * 1024 * 1024   # non-file form fields + multipart framing
MAX_HOLES = 300        # a ski drilled ten times over has well under 100
MAX_TAPE_MARKS = 12    # the UI offers 4
MAX_COORD = 1_000_000  # px / mm sanity bound for any client-supplied coordinate

# Only formats a phone or a browser canvas produces. OpenCV can decode far more
# (TIFF, JPEG 2000, OpenEXR, …) and each extra codec is parser attack surface.
_IMAGE_SIGNATURES = (
    ("JPEG", lambda d: d[:3] == b"\xff\xd8\xff"),
    ("PNG",  lambda d: d[:8] == b"\x89PNG\r\n\x1a\n"),
    ("WEBP", lambda d: d[:4] == b"RIFF" and d[8:12] == b"WEBP"),
)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    # index.html uses inline on*="" handlers and style="" attributes, hence the
    # two *-attr / 'unsafe-inline' allowances; inline <script> blocks stay blocked.
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob: data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; script-src-attr 'unsafe-inline'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
}


class _GuardMiddleware:
    """Pure-ASGI middleware: caps request body size and adds security headers.

    The body cap has to sit in front of the multipart parser — by the time an
    endpoint sees an UploadFile the whole body has already been spooled to /tmp.
    """

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for k, v in _SECURITY_HEADERS.items():
                    headers.setdefault(k, v)
            await send(message)

        detail = f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_body_bytes:
            response = JSONResponse({"detail": detail}, status_code=413)
            await response(scope, receive, send_with_headers)
            return

        received = 0

        async def limited_receive():
            # Covers chunked uploads and clients that lie about Content-Length.
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(413, detail)
            return message

        await self.app(scope, limited_receive, send_with_headers)


# Interactive API docs are off: they are a ready-made map of an unauthenticated API.
app = FastAPI(
    title="Ski Binding Conflict Detector", version="1.0.0",
    docs_url=None, redoc_url=None, openapi_url=None,
)
app.add_middleware(_GuardMiddleware, max_body_bytes=MAX_UPLOAD_BYTES + MAX_FORM_OVERHEAD_BYTES)

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
# Input validation helpers
# ---------------------------------------------------------------------------

def _internal_error(exc: Exception) -> HTTPException:
    """Log the real error server-side; tell the client nothing about internals."""
    log.exception("Unhandled error: %s", exc)
    return HTTPException(status_code=500, detail="Internal error — see the server log")


def _check_image(data: bytes) -> None:
    """Reject anything that is not a reasonably sized JPEG / PNG / WebP.

    Dimensions are read from the file header *before* any bitmap is allocated:
    a 200 kB PNG can declare 14000×14000 px and would otherwise cost >1 GB to decode.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    kind = next((name for name, matches in _IMAGE_SIGNATURES if matches(data)), None)
    if kind is None:
        raise HTTPException(415, "Unsupported image type — please upload a JPEG, PNG or WebP")
    try:
        with Image.open(io.BytesIO(data), formats=[kind]) as im:
            width, height = im.size
    except Image.DecompressionBombError:
        raise HTTPException(413, "Image has too many pixels")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            413,
            f"Image is {width * height / 1e6:.0f} MP; the limit is {MAX_IMAGE_PIXELS / 1e6:.0f} MP",
        )


def _parse_points(raw_json: str, required: tuple[str, ...], what: str,
                  max_items: int, optional: tuple[str, ...] = ()) -> list[dict]:
    """Parse a client-supplied JSON list of point dicts into finite, bounded floats."""
    try:
        raw = json.loads(raw_json)
    except (ValueError, RecursionError):
        raise HTTPException(400, f"{what}: invalid JSON")
    if not isinstance(raw, list):
        raise HTTPException(400, f"{what}: expected a list")
    if len(raw) > max_items:
        raise HTTPException(400, f"{what}: too many entries (limit {max_items})")
    points = []
    for item in raw:
        if not isinstance(item, dict):
            raise HTTPException(400, f"{what}: each entry must be an object")
        try:
            point = {k: float(item[k]) for k in required}
            point.update({k: float(item[k]) for k in optional if item.get(k) is not None})
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, f"{what}: entries need numeric {', '.join(required)}")
        if not all(math.isfinite(v) and abs(v) <= MAX_COORD for v in point.values()):
            raise HTTPException(400, f"{what}: coordinate out of range")
        points.append(point)
    return points


def _unit_axis(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-6:
        raise HTTPException(400, "Ski axis direction is undefined")
    return dx / length, dy / length


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
def list_bindings(category: str = Query("all", max_length=32)):
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
def get_template(
    binding_id: str,
    bsl: float = Query(300.0, ge=100, le=500, allow_inf_nan=False),
    variant: Optional[str] = Query(None, max_length=100),
):
    binding = next(
        (b for b in _binding_db.get("bindings", []) if b["id"] == binding_id), None
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="Binding not found")

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
    # real-world spacing between consecutive marks
    tape_spacing_mm: float = Form(100.0, gt=0, le=5000, allow_inf_nan=False),
    mounting_point_x_px: float = Form(..., ge=-MAX_COORD, le=MAX_COORD, allow_inf_nan=False),
    mounting_point_y_px: float = Form(..., ge=-MAX_COORD, le=MAX_COORD, allow_inf_nan=False),
    axis_dx: float = Form(-1.0, ge=-1.01, le=1.01, allow_inf_nan=False),
    axis_dy: float = Form(0.0, ge=-1.01, le=1.01, allow_inf_nan=False),
):
    """
    Compute scale + optional 1-D perspective correction from tape marks.
    Returns a rectified JPEG (or original if < 3 marks) and calibration params.
    """
    try:
        # Cheap checks first, so a bad request never costs an image decode.
        raw_pts = _parse_points(tape_points_json, ("x_px", "y_px"), "tape_points_json", MAX_TAPE_MARKS)
        tape_points = [(p["x_px"], p["y_px"]) for p in raw_pts]
        n = len(tape_points)
        if n < 2:
            raise HTTPException(400, "Need at least 2 tape marks")

        mpx, mpy = float(mounting_point_x_px), float(mounting_point_y_px)
        adx, ady = _unit_axis(axis_dx, axis_dy)

        image_bytes = await image.read()
        _check_image(image_bytes)
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")

        # Scale from mean tape interval
        u_obs = sorted(
            (x - mpx) * adx + (y - mpy) * ady for x, y in tape_points
        )
        spans_px = [abs(u_obs[i + 1] - u_obs[i]) for i in range(n - 1)]
        if min(spans_px) < 1.0:
            raise HTTPException(400, "Tape marks coincide along the ski axis — place them further apart")
        mean_span_px = float(np.mean(spans_px))
        mm_per_pixel = tape_spacing_mm / mean_span_px

        # Perspective correction when 3+ marks available
        perspective_corrected = False
        if n >= 3:
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
        raise _internal_error(e)


# ---------------------------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    bsl_mm: float = Form(..., ge=100, le=500, allow_inf_nan=False),
    category: str = Form("all", max_length=32),
    min_separation_mm: float = Form(14.0, ge=0, le=100, allow_inf_nan=False),
    top_n: int = Form(20, ge=1, le=1000),
    # Multi-BSL probing: also test at bsl_mm ± bsl_test_range in bsl_test_step increments.
    # Set bsl_test_step=0 to test only at the exact bsl_mm.
    # Bounded: the number of probes is range/step per binding.
    bsl_test_step: int = Form(10, ge=0, le=100),
    bsl_test_range: int = Form(30, ge=0, le=100),
    # Manual calibration overrides (optional)
    mm_per_pixel: Optional[float] = Form(None, ge=0.005, le=50, allow_inf_nan=False),
    mounting_point_x_px: Optional[float] = Form(None, ge=-MAX_COORD, le=MAX_COORD, allow_inf_nan=False),
    mounting_point_y_px: Optional[float] = Form(None, ge=-MAX_COORD, le=MAX_COORD, allow_inf_nan=False),
    # Ski axis direction in image space (unit vector from heel to tip).
    # axis_dx=-1, axis_dy=0  →  horizontal ski, tip to the LEFT  (default)
    # axis_dx=0,  axis_dy=-1 →  vertical ski, tip UPWARD
    axis_dx: Optional[float] = Form(None, ge=-1.01, le=1.01, allow_inf_nan=False),
    axis_dy: Optional[float] = Form(None, ge=-1.01, le=1.01, allow_inf_nan=False),
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
        raise _internal_error(e)


async def _analyze_inner(
    image, bsl_mm, category, min_separation_mm, top_n,
    bsl_test_step, bsl_test_range,
    mm_per_pixel, mounting_point_x_px, mounting_point_y_px,
    axis_dx, axis_dy,
    override_holes_json,
):
    # Build calibration from manually provided values
    calibration: Optional[ScaleCalibration] = None
    if mm_per_pixel is not None and mounting_point_x_px is not None and mounting_point_y_px is not None:
        adx, ady = _unit_axis(
            axis_dx if axis_dx is not None else -1.0,
            axis_dy if axis_dy is not None else 0.0,
        )
        calibration = ScaleCalibration(
            mm_per_pixel=mm_per_pixel,
            ski_centerline_y_px=mounting_point_y_px,
            mounting_point_x_px=mounting_point_x_px,
            axis_dx=adx,
            axis_dy=ady,
        )

    # Parse manually provided holes (empty list = fresh ski, still valid).
    # Malformed input is an error: silently treating it as "no holes" would
    # report every binding as mountable.
    holes_px: list[DetectedHole] = []
    if override_holes_json:
        holes_px = [
            DetectedHole(
                x_px=h["x_px"], y_px=h["y_px"],
                radius_px=min(max(h.get("radius_px", 5.0), 1.0), 500.0),
                confidence=1.0, source="manual",
            )
            for h in _parse_points(override_holes_json, ("x_px", "y_px"), "override_holes_json",
                                   MAX_HOLES, optional=("radius_px",))
        ]

    image_bytes = await image.read()
    _check_image(image_bytes)

    calibration_available = calibration is not None
    mounting_point_known = calibration is not None

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
            log.exception("Overlay rendering failed: %s", e)

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
    binding_id: str = Form(..., max_length=100),
    bsl_mm: float = Form(..., ge=100, le=500, allow_inf_nan=False),
    variant_id: Optional[str] = Form(None, max_length=100),
    min_separation_mm: float = Form(14.0, ge=0, le=100, allow_inf_nan=False),
    existing_holes_json: str = Form("[]"),      # JSON array of {x_abs, y_abs}
    existing_holes_px_json: str = Form("[]"),   # JSON array of {x_px, y_px, radius_px}
    mm_per_pixel: float = Form(..., ge=0.005, le=50, allow_inf_nan=False),
    mounting_point_x_px: float = Form(..., ge=-MAX_COORD, le=MAX_COORD, allow_inf_nan=False),
    mounting_point_y_px: float = Form(..., ge=-MAX_COORD, le=MAX_COORD, allow_inf_nan=False),
    axis_dx: float = Form(-1.0, ge=-1.01, le=1.01, allow_inf_nan=False),
    axis_dy: float = Form(0.0, ge=-1.01, le=1.01, allow_inf_nan=False),
    # override automatic heel search
    heel_offset_mm: Optional[float] = Form(None, ge=-200, le=200, allow_inf_nan=False),
    # shift mounting point along ski axis
    mount_offset_mm: float = Form(0.0, ge=-200, le=200, allow_inf_nan=False),
):
    """Re-render the overlay for a specific binding (interactive overlay controls)."""
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
        raise _internal_error(e)


async def _visualize_inner(
    image, binding_id, bsl_mm, variant_id, min_separation_mm,
    existing_holes_json, existing_holes_px_json,
    mm_per_pixel, mounting_point_x_px, mounting_point_y_px,
    axis_dx, axis_dy, heel_offset_mm, mount_offset_mm,
):
    binding = next(
        (b for b in _binding_db.get("bindings", []) if b["id"] == binding_id), None
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="Binding not found")

    existing_holes = _parse_points(existing_holes_json, ("x_abs", "y_abs"),
                                   "existing_holes_json", MAX_HOLES)
    existing_holes_px = _parse_points(existing_holes_px_json, ("x_px", "y_px"),
                                      "existing_holes_px_json", MAX_HOLES, optional=("radius_px",))
    for h in existing_holes_px:
        if "radius_px" in h:
            h["radius_px"] = min(max(h["radius_px"], 1.0), 500.0)
    axis_dx, axis_dy = _unit_axis(axis_dx, axis_dy)

    image_bytes = await image.read()
    _check_image(image_bytes)

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

    try:
        result = check_binding_conflicts(
            binding, bsl_mm, existing_holes,
            mounting_point_x=0.0,
            min_separation_mm=min_separation_mm,
            variant_id=variant_id,
            heel_offset_mm=heel_offset_mm,
        )
    except ValueError as e:   # e.g. BSL outside this binding's range
        raise HTTPException(status_code=400, detail=str(e))

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
