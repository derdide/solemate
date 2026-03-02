"""
binding_matcher.py
Core logic for:
  - computing absolute hole positions from templates
  - classifying holes as REUSABLE / CONFLICT / NEW
  - ranking all bindings by compatibility with existing holes
  - identifying which binding was previously mounted
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRILL_DIAMETER_MM = 4.1          # Standard ski binding screw hole
DEFAULT_MIN_SEPARATION_MM = 14.0  # Min center-to-center: 4.1 + ~5.8 gap + 4.1
REUSE_TOLERANCE_MM = 2.0         # Centers within this → same hole, reusable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class HoleStatus(Enum):
    NEW = "new"           # No existing hole nearby → must drill fresh
    REUSABLE = "reusable" # Existing hole center within REUSE_TOLERANCE → reuse
    CONFLICT = "conflict"  # Existing hole too close but not reusable → problem!


@dataclass
class HoleAnalysis:
    template_hole: dict            # The candidate binding hole dict
    status: HoleStatus
    nearest_existing: Optional[dict]    # Closest existing hole (if any)
    distance_mm: Optional[float]        # Distance to nearest existing center


@dataclass
class BindingConflictResult:
    binding_id: str
    binding_name: str
    bsl_mm: float
    variant_id: Optional[str]
    verified: bool
    holes_analyzed: list[HoleAnalysis]
    n_new_holes: int
    n_reusable: int
    n_conflicts: int
    conflict_score: float          # Lower = better; 0 = perfect reuse
    is_mountable: bool
    heel_offset_mm: float          # Optimal heel longitudinal offset found
    summary: str


@dataclass
class ScaleCalibration:
    mm_per_pixel: float
    ski_centerline_y_px: float   # y pixel of mounting point (ski centerline)
    mounting_point_x_px: float   # x pixel of mounting point
    # Unit vector heel→tip in image pixel space.
    # Default (-1, 0) = horizontal ski, tip to the LEFT.
    # For vertical ski tip UP use (0, -1).
    axis_dx: float = -1.0
    axis_dy: float = 0.0
    reference_points: list = field(default_factory=list)
    reference_length_mm: float = 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(a: dict, b: dict) -> float:
    """Euclidean distance between two holes' absolute positions."""
    return math.sqrt((a["x_abs"] - b["x_abs"]) ** 2 + (a["y_abs"] - b["y_abs"]) ** 2)


# ---------------------------------------------------------------------------
# Template → absolute coordinates
# ---------------------------------------------------------------------------

def compute_absolute_holes(
    binding: dict,
    bsl_mm: float,
    mounting_point_x: float = 0.0,
    variant_id: Optional[str] = None,
) -> list[dict]:
    """
    Convert template-relative hole positions to absolute ski-frame coordinates.

    Coordinate system (output):
      Origin = mounting point (half BSL from ski centre mark).
      X positive toward tip, Y positive toward skier's right. All mm.

    JSON convention: the half_bsl / fixed entries in binding_db.json store x
    with the positive direction toward the heel (PDF drill-template convention).
    _add() negates x to convert to the code's tip-positive ski frame.
    boot_tip entries (Duke/Baron) use x_from_tip directly — those values are
    already negative (toward heel from boot tip) and do not need negation.

    Returns list of dicts with keys:
      x_abs, y_abs, label, unit ("front"|"heel"), template_x, template_y
    """
    ref = binding["mounting_reference"]
    holes: list[dict] = []

    # Validate BSL
    bsl_range = binding.get("bsl_range_mm")
    if bsl_range is not None:
        if not (bsl_range[0] <= bsl_mm <= bsl_range[1]):
            raise ValueError(
                f"BSL {bsl_mm}mm out of range [{bsl_range[0]}, {bsl_range[1]}] "
                f"for '{binding['name']}'"
            )

    def _add(h: dict, unit: str, x_offset: float = 0.0, y_offset: float = 0.0):
        tx = h.get("x", h.get("x_from_tip", 0.0)) + x_offset
        ty = h["y"] + y_offset
        # Negate: JSON x values increase toward the heel (PDF convention),
        # but the code's ski frame uses x+ toward tip.
        holes.append({
            "x_abs": mounting_point_x - tx,
            "y_abs": ty,
            "label": h["label"],
            "unit": unit,
            "template_x": -tx,
            "template_y": ty,
        })

    if ref == "half_bsl":
        # Standard: template coords are directly relative to mounting_point_x
        for h in binding["front_unit"]["holes"]:
            _add(h, "front")
        for h in binding["heel_unit"]["holes"]:
            _add(h, "heel")

    elif ref == "boot_tip":
        # Duke/Baron: holes given relative to boot tip
        # boot_tip_x (in ski frame) = mounting_point_x + bsl_mm / 2
        boot_tip_x = mounting_point_x + bsl_mm / 2.0

        # Select variant
        variants = binding.get("variants") or []
        if variant_id:
            variant = next((v for v in variants if v["variant_id"] == variant_id), None)
        else:
            # Pick the first variant whose BSL range includes bsl_mm
            variant = next(
                (v for v in variants
                 if v["bsl_range_mm"][0] <= bsl_mm <= v["bsl_range_mm"][1]),
                variants[0] if variants else None,
            )
        if variant is None:
            raise ValueError(f"No suitable variant found for BSL {bsl_mm}mm in '{binding['name']}'")

        for h in variant["front_unit"]["holes"]:
            tx = h["x_from_tip"]   # negative value
            ty = h["y"]
            holes.append({
                "x_abs": boot_tip_x + tx,
                "y_abs": ty,
                "label": h["label"],
                "unit": "front",
                "template_x": tx,
                "template_y": ty,
            })
        for h in variant["heel_unit"]["holes"]:
            tx = h["x_from_tip"]
            ty = h["y"]
            holes.append({
                "x_abs": boot_tip_x + tx,
                "y_abs": ty,
                "label": h["label"],
                "unit": "heel",
                "template_x": tx,
                "template_y": ty,
            })

    elif ref == "fixed":
        # Demo/rental: no BSL scaling; template origin = mounting_point_x
        for h in binding["front_unit"]["holes"]:
            _add(h, "front")
        for h in binding["heel_unit"]["holes"]:
            _add(h, "heel")

    else:
        raise ValueError(f"Unknown mounting_reference: {ref!r}")

    return holes


# ---------------------------------------------------------------------------
# Single-hole analysis
# ---------------------------------------------------------------------------

def analyze_hole(
    candidate: dict,
    existing_holes: list[dict],
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
    reuse_tolerance_mm: float = REUSE_TOLERANCE_MM,
) -> HoleAnalysis:
    """Classify one candidate hole against all existing holes."""
    if not existing_holes:
        return HoleAnalysis(candidate, HoleStatus.NEW, None, None)

    nearest = min(existing_holes, key=lambda e: _dist(candidate, e))
    dist = _dist(candidate, nearest)

    if dist <= reuse_tolerance_mm:
        return HoleAnalysis(candidate, HoleStatus.REUSABLE, nearest, dist)
    elif dist < min_separation_mm:
        return HoleAnalysis(candidate, HoleStatus.CONFLICT, nearest, dist)
    else:
        return HoleAnalysis(candidate, HoleStatus.NEW, nearest, dist)


# ---------------------------------------------------------------------------
# Full binding conflict check
# ---------------------------------------------------------------------------

def check_binding_conflicts(
    binding: dict,
    bsl_mm: float,
    existing_holes_mm: list[dict],
    mounting_point_x: float = 0.0,
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
    reuse_tolerance_mm: float = REUSE_TOLERANCE_MM,
    variant_id: Optional[str] = None,
) -> BindingConflictResult:
    """
    Full conflict analysis for one binding at one BSL.
    Searches the heel's adjustment range to find the offset minimising conflicts.
    """
    candidate_holes = compute_absolute_holes(binding, bsl_mm, mounting_point_x, variant_id)
    front_holes = [h for h in candidate_holes if h["unit"] == "front"]
    heel_holes  = [h for h in candidate_holes if h["unit"] == "heel"]

    # For boot_tip bindings, adjustment_range lives in the selected variant's heel_unit
    if binding["mounting_reference"] == "boot_tip" and variant_id:
        variants = binding.get("variants") or []
        _variant = next((v for v in variants if v["variant_id"] == variant_id), None)
        adj_range = (_variant["heel_unit"].get("adjustment_range_mm") or 0.0) if _variant else 0.0
    else:
        adj_range = binding["heel_unit"].get("adjustment_range_mm") or 0.0

    # --- Front unit (no positional adjustment) ---
    front_analyses = [
        analyze_hole(h, existing_holes_mm, min_separation_mm, reuse_tolerance_mm)
        for h in front_holes
    ]

    # --- Heel unit: search for best longitudinal offset ---
    def _score(analyses: list[HoleAnalysis]) -> tuple[int, int]:
        conflicts = sum(1 for a in analyses if a.status == HoleStatus.CONFLICT)
        reusable  = sum(1 for a in analyses if a.status == HoleStatus.REUSABLE)
        return conflicts, reusable

    best_offset = 0.0
    best_heel_analyses = [
        analyze_hole(h, existing_holes_mm, min_separation_mm, reuse_tolerance_mm)
        for h in heel_holes
    ]
    best_c, best_r = _score(best_heel_analyses)

    if adj_range > 0 and (best_c > 0 or best_r < len(heel_holes)):
        steps = int(adj_range * 2)  # 0.5mm steps
        for step in range(-steps, steps + 1):
            offset = step / 2.0
            shifted = [{**h, "x_abs": h["x_abs"] + offset} for h in heel_holes]
            analyses = [
                analyze_hole(s, existing_holes_mm, min_separation_mm, reuse_tolerance_mm)
                for s in shifted
            ]
            c, r = _score(analyses)
            # Better = fewer conflicts, then more reusable, then smaller offset
            if c < best_c or (c == best_c and r > best_r):
                best_c, best_r = c, r
                best_offset = offset
                best_heel_analyses = analyses

    all_analyses = front_analyses + best_heel_analyses

    n_new      = sum(1 for a in all_analyses if a.status == HoleStatus.NEW)
    n_reusable = sum(1 for a in all_analyses if a.status == HoleStatus.REUSABLE)
    n_conflicts= sum(1 for a in all_analyses if a.status == HoleStatus.CONFLICT)

    # Score: conflicts heavily penalised, new holes minor cost
    conflict_score = n_conflicts * 100.0 + n_new * 1.0 - n_reusable * 0.5
    is_mountable = (n_conflicts == 0)

    # Build summary
    parts = []
    if n_conflicts == 0 and n_reusable == len(all_analyses):
        parts.append("Perfect match - all holes reusable")
    elif n_conflicts == 0:
        parts.append(f"Mountable: {n_new} new hole(s), {n_reusable} reusable")
    else:
        parts.append(f"CONFLICT: {n_conflicts} hole(s) too close. {n_new} new, {n_reusable} reusable")
    if best_offset != 0.0:
        parts.append(f"Heel offset: {best_offset:+.1f}mm")
    if not binding.get("verified", True):
        parts.append("[UNVERIFIED TEMPLATE]")

    return BindingConflictResult(
        binding_id=binding["id"],
        binding_name=binding["name"],
        bsl_mm=bsl_mm,
        variant_id=variant_id,
        verified=binding.get("verified", True),
        holes_analyzed=all_analyses,
        n_new_holes=n_new,
        n_reusable=n_reusable,
        n_conflicts=n_conflicts,
        conflict_score=conflict_score,
        is_mountable=is_mountable,
        heel_offset_mm=best_offset,
        summary=". ".join(parts),
    )


# ---------------------------------------------------------------------------
# Rank all bindings
# ---------------------------------------------------------------------------

def rank_all_bindings(
    binding_db: dict,
    bsl_mm: float,
    existing_holes_mm: list[dict],
    mounting_point_x: float = 0.0,
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
    category_filter: str = "all",
    bsl_test_step: int = 0,
    bsl_test_range: int = 0,
) -> list[BindingConflictResult]:
    """
    Run conflict analysis for every binding in the database.

    By default, tests each binding only at bsl_mm.  When bsl_test_step > 0 the
    function also tests at bsl_mm ± bsl_test_range (in bsl_test_step increments),
    staying within each binding's allowed BSL range.  This lets the caller discover
    whether a slightly different BSL position would create fewer conflicts — useful
    when a ski may have been drilled multiple times for boots of varying sole length.

    Returns results sorted: mountable first (by conflict_score, then bsl delta).
    """
    # Build the list of BSL values to probe.
    # Always include the user-supplied bsl_mm; add nearby values when requested.
    if bsl_test_step > 0 and bsl_test_range > 0:
        step = max(1, int(bsl_test_step))
        half = max(step, int(bsl_test_range))
        bsl_candidates = sorted(set(
            bsl_mm + delta
            for delta in range(-half, half + 1, step)
        ))
    else:
        bsl_candidates = [bsl_mm]

    results: list[BindingConflictResult] = []

    for binding in binding_db["bindings"]:
        # Category filter
        if category_filter != "all" and binding.get("category") != category_filter:
            continue

        bsl_range = binding.get("bsl_range_mm")
        variants = binding.get("variants") or []

        for test_bsl in bsl_candidates:
            # Skip BSL values outside this binding's supported range
            if bsl_range is not None and not (bsl_range[0] <= test_bsl <= bsl_range[1]):
                continue

            try:
                if variants and binding["mounting_reference"] == "boot_tip":
                    # Try each applicable size variant at this BSL
                    for variant in variants:
                        v_range = variant.get("bsl_range_mm")
                        if v_range and (v_range[0] <= test_bsl <= v_range[1]):
                            result = check_binding_conflicts(
                                binding, test_bsl, existing_holes_mm,
                                mounting_point_x, min_separation_mm,
                                variant_id=variant["variant_id"],
                            )
                            results.append(result)
                else:
                    result = check_binding_conflicts(
                        binding, test_bsl, existing_holes_mm,
                        mounting_point_x, min_separation_mm,
                    )
                    results.append(result)
            except (ValueError, KeyError):
                # Skip bindings that can't be computed for this BSL
                continue

    # Sort: mountable first, then by conflict_score, then prefer BSL closest to requested
    results.sort(key=lambda r: (
        not r.is_mountable,
        r.conflict_score,
        abs(r.bsl_mm - bsl_mm),
    ))
    return results


# ---------------------------------------------------------------------------
# Pattern identification
# ---------------------------------------------------------------------------

def identify_previous_binding(
    existing_holes_mm: list[dict],
    binding_db: dict,
    bsl_candidates: Optional[list[float]] = None,
) -> list[dict]:
    """
    Try to identify which binding was previously mounted by matching
    existing hole positions against all templates.

    Returns top 5 matches sorted by match_score descending.
    """
    if bsl_candidates is None:
        bsl_candidates = list(range(240, 400, 20))  # 240..380 in 20mm steps

    matches: list[dict] = []

    for binding in binding_db["bindings"]:
        bsl_range = binding.get("bsl_range_mm")
        if bsl_range is None:
            test_bsls = [315]
        else:
            test_bsls = [
                b for b in bsl_candidates if bsl_range[0] <= b <= bsl_range[1]
            ]

        variants = binding.get("variants") or [None]

        for bsl in test_bsls:
            for variant in variants:
                vid = variant["variant_id"] if variant else None
                try:
                    template_holes = compute_absolute_holes(binding, bsl, 0.0, vid)
                except (ValueError, KeyError):
                    continue

                if not template_holes:
                    continue

                # Try small offsets to handle imprecise mounting point location
                best_score = 0.0
                best_offset = (0.0, 0.0)

                for dx in [-5, -2, 0, 2, 5]:
                    for dy in [-3, 0, 3]:
                        matched = sum(
                            1 for th in template_holes
                            if any(
                                _dist(
                                    {"x_abs": th["x_abs"] + dx, "y_abs": th["y_abs"] + dy},
                                    eh,
                                ) <= REUSE_TOLERANCE_MM * 2
                                for eh in existing_holes_mm
                            )
                        )
                        score = matched / len(template_holes)
                        if score > best_score:
                            best_score = score
                            best_offset = (dx, dy)

                if best_score > 0.3:
                    matches.append({
                        "binding_id": binding["id"],
                        "binding_name": binding["name"],
                        "bsl_mm": bsl,
                        "variant_id": vid,
                        "verified": binding.get("verified", True),
                        "match_score": best_score,
                        "holes_matched": int(best_score * len(template_holes)),
                        "holes_total": len(template_holes),
                        "offset": best_offset,
                    })

    matches.sort(key=lambda m: m["match_score"], reverse=True)
    return matches[:5]


# ---------------------------------------------------------------------------
# Coordinate conversion (pixel ↔ mm)
# ---------------------------------------------------------------------------

def pixels_to_ski_mm(
    holes_px: list,  # list of DetectedHole or dicts with x_px, y_px
    calibration: "ScaleCalibration",
) -> list[dict]:
    """
    Convert detected holes from pixel coords to ski mm frame.
    Origin is at mounting point (mounting_point_x_px, ski_centerline_y_px).
    X grows toward tip along the ski axis; Y grows toward skier's right (lateral).

    The ski axis direction in image space is given by calibration.axis_dx / axis_dy
    (unit vector from heel to tip).  Default is (-1, 0) = horizontal, tip left.

    Forward mapping (image → ski):
        vec = (x_px - cx, y_px - cy)
        ski_x = dot(axis, vec) * mpp         # along axis toward tip
        ski_y = dot(perp,  vec) * mpp        # perp = (axis_dy, -axis_dx)  = right
    """
    result = []
    mpp = calibration.mm_per_pixel
    cx = calibration.mounting_point_x_px
    cy = calibration.ski_centerline_y_px
    # Axis unit vector (heel→tip in image space); normalize defensively
    adx = getattr(calibration, "axis_dx", -1.0)
    ady = getattr(calibration, "axis_dy",  0.0)
    alen = math.sqrt(adx * adx + ady * ady)
    if alen > 1e-6:
        adx /= alen
        ady /= alen

    for h in holes_px:
        x_px = h.x_px if hasattr(h, "x_px") else h["x_px"]
        y_px = h.y_px if hasattr(h, "y_px") else h["y_px"]
        vec_x = x_px - cx
        vec_y = y_px - cy
        ski_x_mm = (adx * vec_x + ady * vec_y) * mpp    # along ski toward tip
        ski_y_mm = (ady * vec_x - adx * vec_y) * mpp    # lateral right

        result.append({
            "x_abs": ski_x_mm,
            "y_abs": ski_y_mm,
            "x_px": x_px,
            "y_px": y_px,
            "confidence": h.confidence if hasattr(h, "confidence") else h.get("confidence", 1.0),
            "source": h.source if hasattr(h, "source") else h.get("source", "unknown"),
        })
    return result


# ---------------------------------------------------------------------------
# Database loader
# ---------------------------------------------------------------------------

def load_binding_db(path: str | Path = "templates/binding_db.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
