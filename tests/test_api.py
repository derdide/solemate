"""Regression tests: request limits and matcher basics.

Run with:  pip install -r requirements-dev.txt && pytest
"""
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402
from src.binding_matcher import check_binding_conflicts, compute_absolute_holes  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def _jpeg(w=400, h=300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 200, 200)).save(buf, "JPEG")
    return buf.getvalue()


def _png_header_only(w: int, h: int) -> bytes:
    """A PNG that merely *declares* w×h pixels — enough to test the pre-decode check."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0")) + chunk(b"IEND", b"")


CAL = dict(mm_per_pixel="0.5", mounting_point_x_px="200", mounting_point_y_px="150", axis_dx="-1", axis_dy="0")
TAPE = json.dumps([{"x_px": 50, "y_px": 250}, {"x_px": 150, "y_px": 250}, {"x_px": 250, "y_px": 250}])


def _analyze(client, **overrides):
    data = dict(bsl_mm="310", bsl_test_step="0", bsl_test_range="0", override_holes_json="[]", **CAL)
    data.update(overrides)
    return client.post("/api/analyze", files={"image": ("a.jpg", _jpeg())}, data=data)


def test_happy_path(client):
    r = client.post("/api/calibrate", files={"image": ("a.jpg", _jpeg())},
                    data=dict(tape_points_json=TAPE, mounting_point_x_px="200", mounting_point_y_px="150"))
    assert r.status_code == 200 and r.json()["calibration"]["mm_per_pixel"] == pytest.approx(1.0)
    r = _analyze(client)
    assert r.status_code == 200 and len(r.json()["results"]) > 0


def test_docs_disabled_and_security_headers(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    headers = client.get("/").headers
    assert "content-security-policy" in headers and headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("field,value", [
    ("bsl_test_range", "1000000"),   # used to pin the CPU for minutes and eat GBs
    ("mm_per_pixel", "0"), ("mm_per_pixel", "1e-7"), ("mm_per_pixel", "nan"), ("bsl_mm", "inf"),
])
def test_out_of_range_numbers_rejected(client, field, value):
    assert _analyze(client, **{field: value}).status_code == 422


def test_hole_list_is_bounded_and_validated(client):
    too_many = json.dumps([{"x_px": i, "y_px": 1} for i in range(main.MAX_HOLES + 1)])
    assert _analyze(client, override_holes_json=too_many).status_code == 400
    # malformed holes must be an error, not "fresh ski, everything mountable"
    assert _analyze(client, override_holes_json="[1, 2]").status_code == 400
    assert _analyze(client, override_holes_json="not json").status_code == 400


def test_image_checks(client):
    bomb = _png_header_only(30000, 30000)
    assert len(bomb) < 200
    r = client.post("/api/calibrate", files={"image": ("a.png", bomb)},
                    data=dict(tape_points_json=TAPE, mounting_point_x_px="1", mounting_point_y_px="1"))
    assert r.status_code == 413
    r = client.post("/api/calibrate", files={"image": ("a.tiff", b"II*\x00" + b"\0" * 64)},
                    data=dict(tape_points_json=TAPE, mounting_point_x_px="1", mounting_point_y_px="1"))
    assert r.status_code == 415


def test_oversized_body_rejected(client):
    r = client.post("/api/calibrate", files={"image": ("a.jpg", b"\0" * (main.MAX_UPLOAD_BYTES + 3 * 1024 * 1024))},
                    data=dict(tape_points_json=TAPE, mounting_point_x_px="1", mounting_point_y_px="1"))
    assert r.status_code == 413


def test_errors_do_not_leak_internals(client, monkeypatch):
    monkeypatch.setattr(main, "rank_all_bindings", lambda *a, **k: 1 / 0)
    r = _analyze(client)
    assert r.status_code == 500 and "ZeroDivision" not in r.text


def test_remounting_same_binding_is_fully_reusable():
    binding = next(b for b in main.load_binding_db(main.DB_PATH)["bindings"] if b["id"] == "marker-jester-griffon-squire")
    existing = [{"x_abs": h["x_abs"], "y_abs": h["y_abs"]} for h in compute_absolute_holes(binding, 315)]
    result = check_binding_conflicts(binding, 315, existing)
    assert (result.n_reusable, result.n_conflicts, result.heel_offset_mm) == (8, 0, 0.0)


def test_heel_search_prefers_smallest_offset():
    binding = next(b for b in main.load_binding_db(main.DB_PATH)["bindings"] if b["id"] == "marker-jester-griffon-squire")
    heel = [h for h in compute_absolute_holes(binding, 315) if h["unit"] == "heel"]
    # An old hole 10 mm to the side of a heel hole: too far to reuse, too close to keep.
    # Shifting the heel ≥ 9.8 mm either way clears it; the search must not run to the
    # end of the adjustment range (12.5 mm) when 10 mm is enough.
    existing = [{"x_abs": heel[0]["x_abs"], "y_abs": heel[0]["y_abs"] - 10.0}]
    result = check_binding_conflicts(binding, 315, existing)
    assert result.n_conflicts == 0 and abs(result.heel_offset_mm) == pytest.approx(10.0)
