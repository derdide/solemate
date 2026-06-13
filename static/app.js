/* app.js — Ski Binding Conflict Detector frontend */
'use strict';

// ── Global state ─────────────────────────────────────────────────────────
let _imageFile   = null;
let _imageEl     = null;   // Image() of the original uploaded photo

// Step 1 — calibration canvas state.
// All click coords are in CANVAS NATURAL PIXELS = image pixels.
// _calScale = displayWidth / naturalWidth — used only for sizing visual elements.
let _calMode      = null;   // 'tape' | 'mp' | 'trans1' | 'trans2' | null
let _tapePoints   = [];     // [{x,y}, …] in canvas natural pixels; 2–4 marks
let _mpPoint      = null;   // {x,y} mount point in canvas natural pixels
let _transL1      = null;   // {a:{x,y}, b:{x,y}} heel-side edge line
let _transL2      = null;   // {a:{x,y}, b:{x,y}} tip-side edge line
let _transL1Start = null;
let _transL2Start = null;
let _axisFlipped  = false;
let _calScale     = 1.0;

// Step 2 — holes & parameters
let _calibration       = null;   // {mm_per_pixel, mounting_point_x_px, ski_centerline_y_px, axis_dx, axis_dy}
let _rectifiedImageEl  = null;   // Image() of the rectified photo
let _rectifiedBlob     = null;   // Blob for sending to API calls
let _editedHoles       = [];     // [{x_px, y_px, radius_px, …}]
let _holesScale        = 1.0;    // displayW / rectifiedNaturalWidth

// Step 3 — overlay / results
let _lastResp        = null;
let _selectedResult  = null;
let _overlayHolesMm  = [];
let _overlayHolesPx  = [];
let _reVisTimer      = null;

// Debug grid (shared across step 2 and step 3)
let _gridEnabled   = false;
let _gridSpacingMm = 10;

// Zoom levels (1.0 = fit to panel width)
let _calZoom   = 1.0;
let _holesZoom = 1.0;

// Active pointer tracking — supports pinch-to-zoom + tap detection
const _calPtrs  = new Map();
const _holePtrs = new Map();
let _pinchCalDist0  = null, _pinchCalZoom0  = null;
let _pinchHoleDist0 = null, _pinchHoleZoom0 = null;

// Tap start + floating-crosshair preview position (canvas natural px)
let _calTapStart     = null;
let _calPreviewPos   = null;
let _holesTapStart   = null;
let _holesPreviewPos = null;
let _previewRafId    = null;


// ── File pick / drag-drop ────────────────────────────────────────────────
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  _imageFile = file;
  document.getElementById('drop-label').textContent = file.name;
  document.getElementById('drop-zone').querySelector('.drop-icon').textContent = '✅';

  const url = URL.createObjectURL(file);
  _imageEl = new Image();
  _imageEl.src = url;
  _imageEl.onload = () => { drawCalCanvas(); };
}


// ── Step 1: Calibration canvas ───────────────────────────────────────────
const calCanvas = document.getElementById('cal-canvas');
const calCtx    = calCanvas.getContext('2d');

function drawCalCanvas() {
  if (!_imageEl) return;
  const displayW = calCanvas.offsetWidth || 340;
  const ratio    = _imageEl.naturalHeight / _imageEl.naturalWidth;
  calCanvas.width  = _imageEl.naturalWidth;
  calCanvas.height = _imageEl.naturalHeight;
  calCanvas.style.height = (displayW * ratio) + 'px';
  _calScale = displayW / _imageEl.naturalWidth;

  calCtx.drawImage(_imageEl, 0, 0);

  // Tape marks and connecting line
  if (_tapePoints.length >= 1) {
    // Connecting line between marks
    if (_tapePoints.length >= 2) {
      calCtx.beginPath();
      calCtx.moveTo(_tapePoints[0].x, _tapePoints[0].y);
      for (let i = 1; i < _tapePoints.length; i++) calCtx.lineTo(_tapePoints[i].x, _tapePoints[i].y);
      calCtx.strokeStyle = '#00e5ff';
      calCtx.lineWidth = 2 / _calScale;
      calCtx.setLineDash([8 / _calScale, 4 / _calScale]);
      calCtx.stroke();
      calCtx.setLineDash([]);

      // Scale estimate from first interval
      const d0 = Math.hypot(_tapePoints[1].x - _tapePoints[0].x, _tapePoints[1].y - _tapePoints[0].y);
      const refMm = parseFloat(document.getElementById('ref-mm').value) || 100;
      document.getElementById('cal-status').textContent =
        `${_tapePoints.length} mark(s) · first interval ${d0.toFixed(0)} px = ${refMm} mm → ~${(refMm / d0).toFixed(4)} mm/px` +
        (_tapePoints.length >= 3 ? ' · parallax correction enabled' : '');
    }
    _tapePoints.forEach((pt, i) => drawDot(pt.x, pt.y, '#00e5ff', String(i + 1)));
  }

  // Mount point crosshair
  if (_mpPoint) {
    const arm = 16 / _calScale;
    calCtx.strokeStyle = '#ffd700';
    calCtx.lineWidth   = 3 / _calScale;
    calCtx.lineCap     = 'round';
    calCtx.beginPath(); calCtx.moveTo(_mpPoint.x - arm, _mpPoint.y); calCtx.lineTo(_mpPoint.x + arm, _mpPoint.y); calCtx.stroke();
    calCtx.beginPath(); calCtx.moveTo(_mpPoint.x, _mpPoint.y - arm); calCtx.lineTo(_mpPoint.x, _mpPoint.y + arm); calCtx.stroke();
  }

  // Axis arrow
  if (_mpPoint && (_tapePoints.length >= 2 || (_transL1 && _transL2))) {
    const { axisDx: adx, axisDy: ady } = _computeAxisAndMount();
    drawAxisArrow(_mpPoint.x, _mpPoint.y, adx, ady);
  }

  // Transversal lines L1 / L2
  if (_transL1)      drawTransLine(_transL1, '#ff55ff', 'L1');
  if (_transL1Start) drawDot(_transL1Start.x, _transL1Start.y, '#ff55ff', 'L1…');
  if (_transL2)      drawTransLine(_transL2, '#cc44cc', 'L2');
  if (_transL2Start) drawDot(_transL2Start.x, _transL2Start.y, '#cc44cc', 'L2…');

  if (_transL1 && _transL2) {
    const midA = { x: (_transL1.a.x + _transL1.b.x) / 2, y: (_transL1.a.y + _transL1.b.y) / 2 };
    const midB = { x: (_transL2.a.x + _transL2.b.x) / 2, y: (_transL2.a.y + _transL2.b.y) / 2 };
    const dx = midB.x - midA.x, dy = midB.y - midA.y;
    const len = Math.hypot(dx, dy) || 1;
    const ext = Math.max(calCanvas.width, calCanvas.height);
    calCtx.beginPath();
    calCtx.moveTo(midA.x - (dx / len) * ext, midA.y - (dy / len) * ext);
    calCtx.lineTo(midA.x + (dx / len) * ext, midA.y + (dy / len) * ext);
    calCtx.strokeStyle = 'rgba(255,100,255,0.65)';
    calCtx.lineWidth = 2 / _calScale;
    calCtx.setLineDash([8 / _calScale, 4 / _calScale]);
    calCtx.stroke();
    calCtx.setLineDash([]);
    drawDot(midA.x, midA.y, '#ff55ff', 'A');
    drawDot(midB.x, midB.y, '#ff55ff', 'B');
  }

  // Floating crosshair preview (touch or mouse while button held)
  if (_calPreviewPos && _calMode) {
    const { x, y } = _calPreviewPos;
    calCtx.save();
    calCtx.globalAlpha = 0.72;
    const r = 10 / _calScale;
    if (_calMode === 'tape') {
      calCtx.beginPath();
      calCtx.arc(x, y, r, 0, Math.PI * 2);
      calCtx.fillStyle = '#00e5ff';
      calCtx.fill();
      calCtx.fillStyle = '#000';
      calCtx.font = `bold ${13 / _calScale}px sans-serif`;
      calCtx.fillText(String(_tapePoints.length + 1), x + r + 2 / _calScale, y + 4 / _calScale);
    } else if (_calMode === 'mp') {
      const arm = 16 / _calScale;
      calCtx.strokeStyle = '#ffd700';
      calCtx.lineWidth = 3 / _calScale;
      calCtx.lineCap = 'round';
      calCtx.beginPath(); calCtx.moveTo(x - arm, y); calCtx.lineTo(x + arm, y); calCtx.stroke();
      calCtx.beginPath(); calCtx.moveTo(x, y - arm); calCtx.lineTo(x, y + arm); calCtx.stroke();
    } else if (_calMode === 'trans1' || _calMode === 'trans2') {
      const col = _calMode === 'trans1' ? '#ff55ff' : '#cc44cc';
      calCtx.beginPath();
      calCtx.arc(x, y, r, 0, Math.PI * 2);
      calCtx.fillStyle = col;
      calCtx.fill();
      const startPt = _calMode === 'trans1' ? _transL1Start : _transL2Start;
      if (startPt) {
        calCtx.beginPath();
        calCtx.moveTo(startPt.x, startPt.y);
        calCtx.lineTo(x, y);
        calCtx.strokeStyle = col;
        calCtx.lineWidth = 2 / _calScale;
        calCtx.stroke();
      }
    }
    calCtx.restore();
  }
}

function drawDot(x, y, color, label) {
  const r = 8 / _calScale;
  calCtx.beginPath();
  calCtx.arc(x, y, r, 0, Math.PI * 2);
  calCtx.fillStyle = color;
  calCtx.fill();
  calCtx.fillStyle = '#000';
  calCtx.font = `${14 / _calScale}px sans-serif`;
  calCtx.fillText(label, x + r + 3 / _calScale, y + 5 / _calScale);
}

function drawTransLine(line, color, label) {
  calCtx.beginPath();
  calCtx.moveTo(line.a.x, line.a.y);
  calCtx.lineTo(line.b.x, line.b.y);
  calCtx.strokeStyle = color;
  calCtx.lineWidth = 2 / _calScale;
  calCtx.stroke();
  drawDot(line.a.x, line.a.y, color, '');
  drawDot(line.b.x, line.b.y, color, '');
  drawDot((line.a.x + line.b.x) / 2, (line.a.y + line.b.y) / 2, color, label);
}

function drawAxisArrow(ox, oy, adx, ady) {
  const len = 100 / _calScale;
  const hw  =  12 / _calScale;
  const ex  = ox + adx * len, ey = oy + ady * len;
  calCtx.strokeStyle = '#ff6600';
  calCtx.fillStyle   = '#ff6600';
  calCtx.lineWidth   = 3 / _calScale;
  calCtx.lineCap     = 'round';
  calCtx.beginPath(); calCtx.moveTo(ox, oy); calCtx.lineTo(ex, ey); calCtx.stroke();
  calCtx.beginPath();
  calCtx.moveTo(ex, ey);
  calCtx.lineTo(ex - adx * hw * 1.5 - ady * hw, ey - ady * hw * 1.5 + adx * hw);
  calCtx.lineTo(ex - adx * hw * 1.5 + ady * hw, ey - ady * hw * 1.5 - adx * hw);
  calCtx.closePath(); calCtx.fill();
  calCtx.font = `bold ${12 / _calScale}px sans-serif`;
  calCtx.fillStyle = '#ff6600';
  calCtx.fillText('→TIP', ex + adx * (hw + 2 / _calScale), ey + ady * (hw + 2 / _calScale));
}

function flipAxis() {
  _axisFlipped = !_axisFlipped;
  document.getElementById('btn-flip').classList.toggle('active', _axisFlipped);
  drawCalCanvas();
}

// Canvas pointer handlers — tap detection + pinch zoom + floating crosshair
function _calNatCoords(e) {
  const rect = calCanvas.getBoundingClientRect();
  const yOff = e.pointerType === 'touch' ? -55 * (calCanvas.height / rect.height) : 0;
  return {
    x: (e.clientX - rect.left) * (calCanvas.width  / (rect.width  || 1)),
    y: (e.clientY - rect.top)  * (calCanvas.height / (rect.height || 1)) + yOff,
  };
}

calCanvas.addEventListener('pointerdown', e => {
  _calPtrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (_calPtrs.size === 1) {
    _calTapStart = { x: e.clientX, y: e.clientY };
  } else if (_calPtrs.size === 2) {
    _calTapStart = null; _calPreviewPos = null;
    const pts = [..._calPtrs.values()];
    _pinchCalDist0 = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
    _pinchCalZoom0 = _calZoom;
  }
});

calCanvas.addEventListener('pointermove', e => {
  _calPtrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (_calPtrs.size >= 2 && _pinchCalDist0 !== null) {
    const pts = [..._calPtrs.values()];
    const panelRect = document.getElementById('cal-panel').getBoundingClientRect();
    const focalPt = { x: (pts[0].x + pts[1].x) / 2 - panelRect.left, y: (pts[0].y + pts[1].y) / 2 - panelRect.top };
    setCalZoom(_pinchCalZoom0 * Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y) / _pinchCalDist0, focalPt);
    return;
  }
  if (_calPtrs.size === 1 && _calTapStart && _calMode) {
    _calPreviewPos = _calNatCoords(e);
    if (!_previewRafId) _previewRafId = requestAnimationFrame(() => { _previewRafId = null; drawCalCanvas(); });
  }
});

calCanvas.addEventListener('pointerup', e => {
  const wasSingle = _calPtrs.size === 1 && _calTapStart !== null;
  const tapStart  = _calTapStart;
  _calPtrs.delete(e.pointerId);
  _calPreviewPos = null;
  if (_calPtrs.size < 2) { _pinchCalDist0 = null; _pinchCalZoom0 = null; }
  if (_calPtrs.size === 0) _calTapStart = null;

  if (!wasSingle || !tapStart) { drawCalCanvas(); return; }
  if (e.pointerType !== 'touch' && Math.hypot(e.clientX - tapStart.x, e.clientY - tapStart.y) >= 10) { drawCalCanvas(); return; }
  if (!_calMode) { drawCalCanvas(); return; }

  const { x: natX, y: natY } = _calNatCoords(e);
  const thr = 20 / _calScale;
  const lbl = document.getElementById('cal-mode-label');

  if (_calMode === 'tape') {
    const idx = _tapePoints.findIndex(p => Math.hypot(p.x - natX, p.y - natY) < thr);
    if (idx >= 0) {
      _tapePoints.splice(idx, 1);
      lbl.textContent = `Mark removed — ${_tapePoints.length} tape mark(s) set.`;
    } else {
      _tapePoints.push({ x: natX, y: natY });
      const n = _tapePoints.length;
      if (n >= 4) {
        _calMode = null; updateCalButtons();
        lbl.textContent = '✓ 4 tape marks set. Now click "② Mount point".';
      } else {
        lbl.textContent = `Mark ${n} set — tap next${n >= 2 ? ' or switch tool' : ''} (${4 - n} more for best correction).`;
      }
    }
    _updateCalibrateButton();

  } else if (_calMode === 'mp') {
    if (_mpPoint && Math.hypot(_mpPoint.x - natX, _mpPoint.y - natY) < thr) {
      _mpPoint = null;
      lbl.textContent = 'Mount point removed — tap to set it again.';
    } else {
      _mpPoint = { x: natX, y: natY };
      _calMode = null; updateCalButtons();
      lbl.textContent = '✓ Mounting point set. Click "Calibrate →", or set axis lines ③④.';
    }
    _updateCalibrateButton();

  } else if (_calMode === 'trans1') {
    if (!_transL1Start) {
      if (_transL1 && (Math.hypot(_transL1.a.x - natX, _transL1.a.y - natY) < thr ||
                       Math.hypot(_transL1.b.x - natX, _transL1.b.y - natY) < thr)) {
        _transL1 = null; lbl.textContent = 'L1 removed — tap one ski edge to start again.';
      } else {
        _transL1Start = { x: natX, y: natY }; lbl.textContent = 'L1: now tap the OPPOSITE ski edge.';
      }
    } else {
      _transL1 = { a: _transL1Start, b: { x: natX, y: natY } };
      _transL1Start = null; _calMode = null; updateCalButtons();
      lbl.textContent = '✓ L1 set. Now click "④ Axis L2" (tip side).';
    }

  } else if (_calMode === 'trans2') {
    if (!_transL2Start) {
      if (_transL2 && (Math.hypot(_transL2.a.x - natX, _transL2.a.y - natY) < thr ||
                       Math.hypot(_transL2.b.x - natX, _transL2.b.y - natY) < thr)) {
        _transL2 = null; lbl.textContent = 'L2 removed — tap one ski edge to start again.';
      } else {
        _transL2Start = { x: natX, y: natY }; lbl.textContent = 'L2: now tap the OPPOSITE ski edge.';
      }
    } else {
      _transL2 = { a: _transL2Start, b: { x: natX, y: natY } };
      _transL2Start = null; _calMode = null; updateCalButtons();
      lbl.textContent = '✓ Centreline from L1 & L2. Click "Calibrate →".';
    }
  }
  drawCalCanvas();
});

calCanvas.addEventListener('pointercancel', e => {
  _calPtrs.delete(e.pointerId); _calTapStart = null; _calPreviewPos = null;
  if (_calPtrs.size < 2) { _pinchCalDist0 = null; _pinchCalZoom0 = null; }
  drawCalCanvas();
});

function startCalMode(mode) {
  _calMode = mode;
  updateCalButtons();
  if (mode === 'tape') {
    _tapePoints  = [];
    _axisFlipped = false;
    document.getElementById('btn-flip').classList.remove('active');
    document.getElementById('cal-mode-label').textContent =
      'Click the FIRST tape mark (heel side). Then click 1–3 more marks at equal spacing toward the tip.';
  } else if (mode === 'mp') {
    document.getElementById('cal-mode-label').textContent =
      'Click the ski CENTRE MARK (half-sole / mounting reference line on ski).';
  } else if (mode === 'trans1') {
    _transL1 = null; _transL1Start = null;
    document.getElementById('cal-mode-label').textContent =
      'L1 (heel side): click one ski EDGE, then click the OPPOSITE edge.';
  } else if (mode === 'trans2') {
    _transL2 = null; _transL2Start = null;
    document.getElementById('cal-mode-label').textContent =
      'L2 (tip side): click one ski EDGE, then click the OPPOSITE edge.';
  }
  drawCalCanvas();
}

function updateCalButtons() {
  document.getElementById('btn-tape').classList.toggle('active',   _calMode === 'tape');
  document.getElementById('btn-mp').classList.toggle('active',     _calMode === 'mp');
  document.getElementById('btn-trans1').classList.toggle('active', _calMode === 'trans1');
  document.getElementById('btn-trans2').classList.toggle('active', _calMode === 'trans2');
}

function _updateCalibrateButton() {
  const btn = document.getElementById('calibrate-btn');
  if (btn) btn.disabled = !(_tapePoints.length >= 2 && _mpPoint);
}

// Compute axis direction + (possibly projected) mount point from current calibration state
function _computeAxisAndMount() {
  let axisDx, axisDy;
  let mpX = _mpPoint ? _mpPoint.x : 0;
  let mpY = _mpPoint ? _mpPoint.y : 0;

  if (_transL1 && _transL2) {
    const midA = { x: (_transL1.a.x + _transL1.b.x) / 2, y: (_transL1.a.y + _transL1.b.y) / 2 };
    const midB = { x: (_transL2.a.x + _transL2.b.x) / 2, y: (_transL2.a.y + _transL2.b.y) / 2 };
    const dx = midB.x - midA.x, dy = midB.y - midA.y;
    const len = Math.hypot(dx, dy) || 1;
    axisDx = dx / len; axisDy = dy / len;
    if (_mpPoint) {
      const t = (_mpPoint.x - midA.x) * axisDx + (_mpPoint.y - midA.y) * axisDy;
      mpX = midA.x + t * axisDx;
      mpY = midA.y + t * axisDy;
    }
  } else if (_tapePoints.length >= 2) {
    const first = _tapePoints[0], last = _tapePoints[_tapePoints.length - 1];
    const dx = last.x - first.x, dy = last.y - first.y;
    const len = Math.hypot(dx, dy) || 1;
    axisDx = dx / len; axisDy = dy / len;
  } else {
    axisDx = -1.0; axisDy = 0.0;
  }

  if (_axisFlipped) { axisDx = -axisDx; axisDy = -axisDy; }
  return { axisDx, axisDy, mpX, mpY };
}

window.addEventListener('resize', () => {
  if (_imageEl)           { _calZoom = 1.0; calCanvas.style.width = '';   drawCalCanvas();   _updateZoomLabel('cal-zoom-label', 1.0); }
  if (_rectifiedImageEl)  { _holesZoom = 1.0; holesCanvas.style.width = ''; drawHolesCanvas(); _updateZoomLabel('holes-zoom-label', 1.0); }
});


// ── Step 1: Calibrate action ─────────────────────────────────────────────
async function doCalibrate() {
  if (!_imageFile)              { showError('No image loaded.');                        return; }
  if (_tapePoints.length < 2)  { showError('Click at least 2 tape marks first.');      return; }
  if (!_mpPoint)                { showError('Click the mounting point on the ski.');    return; }

  const refMm = parseFloat(document.getElementById('ref-mm').value);
  if (isNaN(refMm) || refMm <= 0) { showError('Enter a valid mark spacing.'); return; }

  const { axisDx, axisDy, mpX, mpY } = _computeAxisAndMount();
  clearError();
  setLoading(true, 'Calibrating…');

  const fd = new FormData();
  fd.append('image', _imageFile);
  fd.append('tape_points_json', JSON.stringify(_tapePoints.map(p => ({ x_px: p.x, y_px: p.y }))));
  fd.append('tape_spacing_mm', refMm);
  fd.append('mounting_point_x_px', mpX);
  fd.append('mounting_point_y_px', mpY);
  fd.append('axis_dx', axisDx);
  fd.append('axis_dy', axisDy);

  try {
    const resp = await fetch('/api/calibrate', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(`Server error ${resp.status}: ${await resp.text()}`);
    const data = await resp.json();

    _calibration  = data.calibration;
    _editedHoles  = [];
    _rectifiedBlob = base64ToBlob(data.rectified_image_base64, 'image/jpeg');

    _rectifiedImageEl = new Image();
    _rectifiedImageEl.onload = () => {
      const cal = _calibration;
      const corrNote = cal.perspective_corrected ? ' · parallax corrected' : '';
      document.getElementById('cal-summary').textContent =
        `Scale: ${cal.mm_per_pixel.toFixed(4)} mm/px${corrNote}`;
      drawHolesCanvas();
    };
    _rectifiedImageEl.src = URL.createObjectURL(_rectifiedBlob);

    showStep(2);
  } catch (err) {
    showError(err.message || 'Calibration failed.');
  } finally {
    setLoading(false);
  }
}

function base64ToBlob(b64, mime) {
  const binary = atob(b64);
  const arr = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i);
  return new Blob([arr], { type: mime });
}


// ── Step 2: Holes canvas ─────────────────────────────────────────────────
const holesCanvas = document.getElementById('holes-canvas');
const holesCtx    = holesCanvas.getContext('2d');

function drawHolesCanvas() {
  if (!_rectifiedImageEl) return;
  const displayW = holesCanvas.offsetWidth || 340;
  const ratio    = _rectifiedImageEl.naturalHeight / _rectifiedImageEl.naturalWidth;
  holesCanvas.width  = _rectifiedImageEl.naturalWidth;
  holesCanvas.height = _rectifiedImageEl.naturalHeight;
  holesCanvas.style.height = (displayW * ratio) + 'px';
  _holesScale = displayW / _rectifiedImageEl.naturalWidth;

  holesCtx.drawImage(_rectifiedImageEl, 0, 0);

  // Debug grid
  if (_gridEnabled && _calibration) {
    const cal = _calibration;
    _drawGrid(holesCtx, holesCanvas.width, holesCanvas.height,
      cal.mounting_point_x_px, cal.ski_centerline_y_px,
      cal.axis_dx, cal.axis_dy, cal.mm_per_pixel);
  }

  // Mount point crosshair
  if (_calibration) {
    const { mounting_point_x_px: mpx, ski_centerline_y_px: mpy } = _calibration;
    const arm = 16 / _holesScale;
    holesCtx.strokeStyle = '#ffd700';
    holesCtx.lineWidth   = 3 / _holesScale;
    holesCtx.lineCap     = 'round';
    holesCtx.beginPath(); holesCtx.moveTo(mpx - arm, mpy); holesCtx.lineTo(mpx + arm, mpy); holesCtx.stroke();
    holesCtx.beginPath(); holesCtx.moveTo(mpx, mpy - arm); holesCtx.lineTo(mpx, mpy + arm); holesCtx.stroke();
  }

  // Holes
  (_editedHoles || []).forEach(h => {
    const r = Math.max(h.radius_px || 5, 15 / _holesScale);
    holesCtx.beginPath();
    holesCtx.arc(h.x_px, h.y_px, r, 0, Math.PI * 2);
    holesCtx.strokeStyle = 'rgba(50,255,100,0.9)';
    holesCtx.lineWidth = 3 / _holesScale;
    holesCtx.stroke();
  });

  // Floating crosshair preview for holes
  if (_holesPreviewPos) {
    const { x, y } = _holesPreviewPos;
    const r = Math.max(15 / _holesScale, 12);
    holesCtx.save();
    holesCtx.globalAlpha = 0.65;
    holesCtx.beginPath();
    holesCtx.arc(x, y, r, 0, Math.PI * 2);
    holesCtx.strokeStyle = 'rgba(50,255,100,0.9)';
    holesCtx.lineWidth = 3 / _holesScale;
    holesCtx.setLineDash([4 / _holesScale, 4 / _holesScale]);
    holesCtx.stroke();
    holesCtx.setLineDash([]);
    const arm = r * 0.6;
    holesCtx.strokeStyle = 'rgba(50,255,100,0.9)';
    holesCtx.lineWidth = 2 / _holesScale;
    holesCtx.beginPath(); holesCtx.moveTo(x - arm, y); holesCtx.lineTo(x + arm, y); holesCtx.stroke();
    holesCtx.beginPath(); holesCtx.moveTo(x, y - arm); holesCtx.lineTo(x, y + arm); holesCtx.stroke();
    holesCtx.restore();
  }

  // Count label
  const n = (_editedHoles || []).length;
  const lbl = document.getElementById('holes-count-label');
  if (lbl) lbl.textContent = n === 0 ? 'No holes marked' : `${n} hole(s)`;
}

function _holeNatCoords(e) {
  const rect = holesCanvas.getBoundingClientRect();
  const yOff = e.pointerType === 'touch' ? -55 * (holesCanvas.height / rect.height) : 0;
  return {
    x: (e.clientX - rect.left) * (holesCanvas.width  / (rect.width  || 1)),
    y: (e.clientY - rect.top)  * (holesCanvas.height / (rect.height || 1)) + yOff,
  };
}

holesCanvas.addEventListener('pointerdown', e => {
  _holePtrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (_holePtrs.size === 1) {
    _holesTapStart = { x: e.clientX, y: e.clientY };
  } else if (_holePtrs.size === 2) {
    _holesTapStart = null; _holesPreviewPos = null;
    const pts = [..._holePtrs.values()];
    _pinchHoleDist0 = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
    _pinchHoleZoom0 = _holesZoom;
  }
});

holesCanvas.addEventListener('pointermove', e => {
  _holePtrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (_holePtrs.size >= 2 && _pinchHoleDist0 !== null) {
    const pts = [..._holePtrs.values()];
    const panelRect = document.getElementById('holes-panel').getBoundingClientRect();
    const focalPt = { x: (pts[0].x + pts[1].x) / 2 - panelRect.left, y: (pts[0].y + pts[1].y) / 2 - panelRect.top };
    setHolesZoom(_pinchHoleZoom0 * Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y) / _pinchHoleDist0, focalPt);
    return;
  }
  if (_holePtrs.size === 1 && _holesTapStart && _rectifiedImageEl) {
    _holesPreviewPos = _holeNatCoords(e);
    if (!_previewRafId) _previewRafId = requestAnimationFrame(() => { _previewRafId = null; drawHolesCanvas(); });
  }
});

holesCanvas.addEventListener('pointerup', e => {
  const wasSingle = _holePtrs.size === 1 && _holesTapStart !== null;
  const tapStart  = _holesTapStart;
  _holePtrs.delete(e.pointerId);
  _holesPreviewPos = null;
  if (_holePtrs.size < 2) { _pinchHoleDist0 = null; _pinchHoleZoom0 = null; }
  if (_holePtrs.size === 0) _holesTapStart = null;

  if (!wasSingle || !tapStart || !_rectifiedImageEl) { drawHolesCanvas(); return; }
  if (e.pointerType !== 'touch' && Math.hypot(e.clientX - tapStart.x, e.clientY - tapStart.y) >= 10) { drawHolesCanvas(); return; }

  const { x: natX, y: natY } = _holeNatCoords(e);
  if (_editedHoles === null) _editedHoles = [];
  const threshold = 20 / _holesScale;
  const idx = _editedHoles.findIndex(h => Math.hypot(h.x_px - natX, h.y_px - natY) < threshold);
  if (idx >= 0) {
    _editedHoles.splice(idx, 1);
  } else {
    _editedHoles.push({ x_px: Math.round(natX), y_px: Math.round(natY), radius_px: 5, confidence: 1.0, source: 'manual' });
  }
  drawHolesCanvas();
});

holesCanvas.addEventListener('pointercancel', e => {
  _holePtrs.delete(e.pointerId); _holesTapStart = null; _holesPreviewPos = null;
  if (_holePtrs.size < 2) { _pinchHoleDist0 = null; _pinchHoleZoom0 = null; }
  drawHolesCanvas();
});

function clearAllHoles() {
  _editedHoles = [];
  drawHolesCanvas();
}

function backToCalibration() {
  if (_editedHoles && _editedHoles.length > 0) {
    if (!confirm('Going back will clear all hole marks. Continue?')) return;
  }
  _editedHoles       = [];
  _rectifiedImageEl  = null;
  _rectifiedBlob     = null;
  _calibration       = null;
  showStep(1);
}

// (resize handled above)


// ── Step 2: Analyze action ───────────────────────────────────────────────
async function doAnalyze() {
  if (!_rectifiedBlob || !_calibration) { showError('Complete calibration first.'); return; }

  const cal = _calibration;
  clearError();
  setLoading(true, 'Analyzing…');

  const fd = new FormData();
  fd.append('image', _rectifiedBlob, 'ski_rectified.jpg');
  fd.append('bsl_mm',              document.getElementById('bsl-input').value);
  fd.append('category',            document.getElementById('cat-select').value);
  fd.append('min_separation_mm',   document.getElementById('sep-input').value);
  fd.append('top_n',               999);
  fd.append('bsl_test_step',       0);
  fd.append('bsl_test_range',      0);
  fd.append('mm_per_pixel',        cal.mm_per_pixel);
  fd.append('mounting_point_x_px', cal.mounting_point_x_px);
  fd.append('mounting_point_y_px', cal.ski_centerline_y_px);
  fd.append('axis_dx',             cal.axis_dx);
  fd.append('axis_dy',             cal.axis_dy);
  fd.append('override_holes_json', JSON.stringify(_editedHoles || []));

  try {
    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(`Server error ${resp.status}: ${await resp.text()}`);
    _lastResp = await resp.json();
    renderResults(_lastResp);
  } catch (err) {
    showError(err.message || 'Analysis failed.');
  } finally {
    setLoading(false);
  }
}


// ── Step 3: Render results ───────────────────────────────────────────────
function renderResults(data) {
  showStep(3);

  // Overlay image
  const overlayImg       = document.getElementById('overlay-img');
  const overlayContainer = document.getElementById('overlay-img-container');
  const previewCvs       = document.getElementById('preview-canvas');

  if (data.output_image_base64) {
    overlayImg.src = 'data:image/jpeg;base64,' + data.output_image_base64;
    overlayContainer.style.display = 'block';
    previewCvs.style.display = 'none';
  } else {
    overlayContainer.style.display = 'none';
    previewCvs.style.display = 'block';
  }

  // Hole count caption
  document.getElementById('hole-count').textContent =
    `${data.detected_holes} hole(s) marked · scale ${data.calibration.mm_per_pixel?.toFixed(4)} mm/px`;

  // Results table
  const tableCard = document.getElementById('results-table-card');
  const tbody     = document.getElementById('results-body');
  tbody.innerHTML = '';

  if (!data.mounting_point_known || !data.results || data.results.length === 0) {
    tableCard.style.display = data.mounting_point_known ? 'block' : 'none';
    if (data.mounting_point_known && data.results.length === 0) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="7" style="text-align:center;color:var(--text-dim)">No compatible bindings found for this BSL.</td>';
      tbody.appendChild(tr);
      tableCard.style.display = 'block';
    }
    return;
  }

  tableCard.style.display = 'block';
  document.getElementById('img-binding-label').textContent = `— ${data.results[0].binding_name}`;
  const enteredBsl = parseFloat(document.getElementById('bsl-input').value) || 0;

  data.results.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.className = r.is_mountable ? 'mountable' : '';
    if (idx === 0) tr.classList.add('selected');

    const heelOff    = r.heel_offset_mm !== 0 ? `${r.heel_offset_mm > 0 ? '+' : ''}${r.heel_offset_mm.toFixed(1)}mm` : '—';
    const statusBadge = r.is_mountable
      ? '<span class="badge badge-ok">OK</span>'
      : '<span class="badge badge-err">CONFLICT</span>';
    const unverified = r.verified ? '' : ' <span class="badge badge-warn" title="Unverified template">⚠</span>';
    const varLabel   = r.variant_id ? ` <small style="color:var(--text-dim)">[${r.variant_id}]</small>` : '';
    const bslDiff    = r.bsl_mm - enteredBsl;
    const bslStyle   = bslDiff !== 0 ? 'font-weight:bold;color:var(--accent)' : 'color:var(--text-dim)';
    const bslLabel   = bslDiff === 0 ? `${r.bsl_mm}`
      : `${r.bsl_mm} <small style="font-weight:normal">(${bslDiff > 0 ? '+' : ''}${bslDiff})</small>`;

    tr.innerHTML = `
      <td>${r.binding_name}${varLabel}${unverified}</td>
      <td style="${bslStyle};font-size:0.85rem">${bslLabel}</td>
      <td>${r.n_new_holes}</td>
      <td>${r.n_reusable}</td>
      <td>${r.n_conflicts}</td>
      <td style="font-size:0.8rem">${heelOff}</td>
      <td>${statusBadge}</td>`;

    tr.addEventListener('click', () => selectBinding(tr, r));
    tbody.appendChild(tr);
  });

  // Auto-select first result
  const firstTr = tbody.querySelector('tr');
  if (firstTr) selectBinding(firstTr, data.results[0]);
}


// ── Step 3: Select binding row → init controls + visualize ───────────────
function selectBinding(tr, result) {
  document.querySelectorAll('#results-body tr').forEach(r => r.classList.remove('selected'));
  tr.classList.add('selected');
  document.getElementById('img-binding-label').textContent = `— ${result.binding_name}`;
  if (!_lastResp?.mounting_point_known) return;

  _selectedResult = result;

  const cal = _lastResp.calibration;
  const adx = cal.axis_dx ?? -1.0;
  const ady = cal.axis_dy ??  0.0;
  _overlayHolesMm = (_lastResp.holes_px || []).map(h => {
    const vx = h.x_px - cal.mounting_point_x_px;
    const vy = h.y_px - cal.ski_centerline_y_px;
    return { x_abs: (adx * vx + ady * vy) * cal.mm_per_pixel, y_abs: (ady * vx - adx * vy) * cal.mm_per_pixel };
  });
  _overlayHolesPx = (_lastResp.holes_px || []).map(h => ({
    x_px: h.x_px, y_px: h.y_px, radius_px: h.radius_px,
  }));

  const bslRange = result.bsl_range_mm;
  const bslMin = bslRange ? bslRange[0] : Math.max(240, result.bsl_mm - 30);
  const bslMax = bslRange ? bslRange[1] : Math.min(400, result.bsl_mm + 30);
  _setSlider('ctrl-bsl', result.bsl_mm, bslMin, bslMax, 1);

  const adjRange = result.adjustment_range_mm || 0;
  const heelGroup = document.getElementById('ctrl-heel-group');
  if (adjRange > 0) {
    heelGroup.style.display = '';
    _setSlider('ctrl-heel', result.heel_offset_mm || 0, -adjRange, adjRange, 0.5);
  } else {
    heelGroup.style.display = 'none';
  }
  _setSlider('ctrl-mount', 0, -30, 30, 0.5);

  document.getElementById('overlay-controls-card').style.display = '';
  _scheduleReVisualize(0);
}

function _setSlider(id, value, min, max, step) {
  const rng = document.getElementById(id);
  const num = document.getElementById(id + '-num');
  rng.min = min; rng.max = max; rng.step = step; rng.value = value;
  num.min = min; num.max = max; num.step = step; num.value = value;
}

['ctrl-bsl', 'ctrl-heel', 'ctrl-mount'].forEach(id => {
  const rng = document.getElementById(id);
  const num = document.getElementById(id + '-num');
  if (!rng || !num) return;
  rng.addEventListener('input', () => {
    num.value = rng.value;
    if (id === 'ctrl-mount') _drawOverlayCanvas();
    _scheduleReVisualize(80);
  });
  num.addEventListener('input', () => {
    rng.value = num.value;
    if (id === 'ctrl-mount') _drawOverlayCanvas();
    _scheduleReVisualize(80);
  });
});

function _scheduleReVisualize(delayMs) {
  clearTimeout(_reVisTimer);
  _reVisTimer = setTimeout(_reVisualize, delayMs);
}

async function _reVisualize() {
  if (!_selectedResult || !_lastResp || !_rectifiedBlob) return;
  const cal = _lastResp.calibration;

  const bsl        = parseFloat(document.getElementById('ctrl-bsl').value)   || _selectedResult.bsl_mm;
  const heelOffset = parseFloat(document.getElementById('ctrl-heel').value)  || 0;
  const mountOff   = parseFloat(document.getElementById('ctrl-mount').value) || 0;

  const fd = new FormData();
  fd.append('image',                  _rectifiedBlob, 'ski_rectified.jpg');
  fd.append('binding_id',             _selectedResult.binding_id);
  fd.append('bsl_mm',                 bsl);
  fd.append('variant_id',             _selectedResult.variant_id || '');
  fd.append('min_separation_mm',      document.getElementById('sep-input').value);
  fd.append('existing_holes_json',    JSON.stringify(_overlayHolesMm));
  fd.append('existing_holes_px_json', JSON.stringify(_overlayHolesPx));
  fd.append('mm_per_pixel',           cal.mm_per_pixel);
  fd.append('mounting_point_x_px',    cal.mounting_point_x_px);
  fd.append('mounting_point_y_px',    cal.ski_centerline_y_px);
  fd.append('axis_dx',                cal.axis_dx ?? -1.0);
  fd.append('axis_dy',                cal.axis_dy ??  0.0);
  fd.append('heel_offset_mm',         heelOffset);
  fd.append('mount_offset_mm',        mountOff);

  try {
    const resp = await fetch('/api/visualize', { method: 'POST', body: fd });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.output_image_base64) {
      document.getElementById('overlay-img').src = 'data:image/jpeg;base64,' + data.output_image_base64;
      document.getElementById('overlay-img-container').style.display = 'block';
      document.getElementById('preview-canvas').style.display = 'none';
      _drawOverlayCanvas();
    }
  } catch (_) {}
}


// ── Overlay canvas (step 3 crosses + grid) ───────────────────────────────
function _drawOverlayCanvas() {
  if (!_selectedResult || !_lastResp || !_rectifiedImageEl) return;
  const container = document.getElementById('overlay-img-container');
  if (!container || container.style.display === 'none') return;

  const canvas = document.getElementById('overlay-canvas');
  const cal    = _lastResp.calibration;
  const natW   = _rectifiedImageEl.naturalWidth;
  const natH   = _rectifiedImageEl.naturalHeight;
  canvas.width  = natW;
  canvas.height = natH;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, natW, natH);

  const mpx = cal.mounting_point_x_px;
  const mpy = cal.ski_centerline_y_px;
  const adx = cal.axis_dx ?? -1.0;
  const ady = cal.axis_dy ??  0.0;
  const mpp = cal.mm_per_pixel;

  const mountOff = parseFloat(document.getElementById('ctrl-mount').value) || 0;
  const offPx    = mountOff / mpp;
  const offX     = mpx + offPx * adx;
  const offY     = mpy + offPx * ady;

  const armPx = Math.max(20, natW * 0.014);
  const lw    = Math.max(2,  natW * 0.0018);

  if (_gridEnabled) _drawGrid(ctx, natW, natH, mpx, mpy, adx, ady, mpp);

  // Fixed yellow cross at original calibration mount point
  _drawCross(ctx, mpx, mpy, armPx, lw, '#ffd700', null);

  // Moving white/black cross at offset position (only when offset is non-zero)
  if (Math.abs(mountOff) >= 0.1) {
    _drawCross(ctx, offX, offY, armPx * 0.85, lw, '#ffffff', '#000000');
  }
}

function _drawCross(ctx, x, y, arm, lw, color, borderColor) {
  ctx.lineCap = 'round';
  if (borderColor) {
    ctx.strokeStyle = borderColor;
    ctx.lineWidth   = lw * 3.5;
    ctx.beginPath(); ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm); ctx.stroke();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth   = lw;
  ctx.beginPath(); ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm); ctx.stroke();
}

function _drawGrid(ctx, natW, natH, mpx, mpy, adx, ady, mpp) {
  const spacingPx = _gridSpacingMm / mpp;
  const pdx = -ady;
  const pdy =  adx;
  const diagLen = Math.hypot(natW, natH) + spacingPx * 2;
  const nLines  = Math.ceil(diagLen / spacingPx) + 1;
  const lw = Math.max(1, natW * 0.0008);

  ctx.save();
  ctx.lineWidth = lw;
  ctx.setLineDash([natW * 0.004, natW * 0.004]);

  for (let i = -nLines; i <= nLines; i++) {
    ctx.strokeStyle = i === 0 ? 'rgba(255,255,100,0.45)' : 'rgba(255,255,255,0.18)';
    // Lines parallel to axis at lateral offset i * spacingPx
    const ox1 = mpx + i * spacingPx * pdx, oy1 = mpy + i * spacingPx * pdy;
    ctx.beginPath();
    ctx.moveTo(ox1 - adx * diagLen, oy1 - ady * diagLen);
    ctx.lineTo(ox1 + adx * diagLen, oy1 + ady * diagLen);
    ctx.stroke();
    // Lines perpendicular to axis at along-axis offset i * spacingPx
    const ox2 = mpx + i * spacingPx * adx, oy2 = mpy + i * spacingPx * ady;
    ctx.beginPath();
    ctx.moveTo(ox2 - pdx * diagLen, oy2 - pdy * diagLen);
    ctx.lineTo(ox2 + pdx * diagLen, oy2 + pdy * diagLen);
    ctx.stroke();
  }
  ctx.restore();
}

function toggleGrid() {
  _gridEnabled = !_gridEnabled;
  document.querySelectorAll('.btn-grid-toggle').forEach(btn => {
    btn.textContent = _gridEnabled ? 'Grid on' : 'Grid off';
    btn.classList.toggle('active', _gridEnabled);
  });
  drawHolesCanvas();
  _drawOverlayCanvas();
}

function setGridSpacing(mm) {
  _gridSpacingMm = mm;
  document.querySelectorAll('.btn-grid-1cm').forEach(btn => btn.classList.toggle('active', mm === 10));
  document.querySelectorAll('.btn-grid-5mm').forEach(btn => btn.classList.toggle('active', mm === 5));
  drawHolesCanvas();
  _drawOverlayCanvas();
}


// ── Zoom ─────────────────────────────────────────────────────────────────
function _updateZoomLabel(id, zoom) {
  const el = document.getElementById(id);
  if (el) el.textContent = Math.round(zoom * 100) + '%';
}

function setCalZoom(level, focalPt) {
  const panel = document.getElementById('cal-panel');
  const oldZoom = _calZoom;
  const baseW = oldZoom > 1.0 ? parseFloat(calCanvas.style.width) / oldZoom : calCanvas.offsetWidth;
  _calZoom = Math.max(1.0, Math.min(8.0, level));
  calCanvas.style.width = _calZoom > 1.0 ? (baseW * _calZoom) + 'px' : '';
  _updateZoomLabel('cal-zoom-label', _calZoom);
  if (focalPt && oldZoom !== _calZoom) {
    const r = _calZoom / oldZoom;
    panel.scrollLeft = (panel.scrollLeft + focalPt.x) * r - focalPt.x;
    panel.scrollTop  = (panel.scrollTop  + focalPt.y) * r - focalPt.y;
  }
  if (_imageEl) drawCalCanvas();
}

function setHolesZoom(level, focalPt) {
  const panel = document.getElementById('holes-panel');
  const oldZoom = _holesZoom;
  const baseW = oldZoom > 1.0 ? parseFloat(holesCanvas.style.width) / oldZoom : holesCanvas.offsetWidth;
  _holesZoom = Math.max(1.0, Math.min(8.0, level));
  holesCanvas.style.width = _holesZoom > 1.0 ? (baseW * _holesZoom) + 'px' : '';
  _updateZoomLabel('holes-zoom-label', _holesZoom);
  if (focalPt && oldZoom !== _holesZoom) {
    const r = _holesZoom / oldZoom;
    panel.scrollLeft = (panel.scrollLeft + focalPt.x) * r - focalPt.x;
    panel.scrollTop  = (panel.scrollTop  + focalPt.y) * r - focalPt.y;
  }
  if (_rectifiedImageEl) drawHolesCanvas();
}

function adjustZoom(which, dir) {
  const factor = dir > 0 ? 1.3 : 1 / 1.3;
  if (which === 'cal') setCalZoom(_calZoom * factor);
  else setHolesZoom(_holesZoom * factor);
}

function resetZoom(which) {
  if (which === 'cal') setCalZoom(1.0);
  else setHolesZoom(1.0);
}

// Wheel zoom on canvas panels — zoom centred on cursor position
document.getElementById('cal-panel').addEventListener('wheel', e => {
  if (!_imageEl) return;
  e.preventDefault();
  const rect = document.getElementById('cal-panel').getBoundingClientRect();
  const focalPt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  setCalZoom(_calZoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), focalPt);
}, { passive: false });

document.getElementById('holes-panel').addEventListener('wheel', e => {
  if (!_rectifiedImageEl) return;
  e.preventDefault();
  const rect = document.getElementById('holes-panel').getBoundingClientRect();
  const focalPt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  setHolesZoom(_holesZoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), focalPt);
}, { passive: false });


// ── Step navigation ──────────────────────────────────────────────────────
function updateStepNav(n) {
  for (let i = 1; i <= 3; i++) {
    const el = document.getElementById('snav-' + i);
    el.classList.remove('active', 'done');
    if (i < n)      el.classList.add('done');
    else if (i === n) el.classList.add('active');
  }
}

function showStep(n) {
  document.getElementById('step1').style.display = n === 1 ? '' : 'none';
  document.getElementById('step2').style.display = n === 2 ? '' : 'none';
  document.getElementById('step3').style.display = n === 3 ? '' : 'none';
  updateStepNav(n);
  if (n === 1) drawCalCanvas();
  if (n === 2) drawHolesCanvas();
}

// Step nav click — done steps allow going back; active/future steps are non-interactive
[1, 2, 3].forEach(n => {
  document.getElementById('snav-' + n).addEventListener('click', () => {
    const el = document.getElementById('snav-' + n);
    if (el.classList.contains('done')) showStep(n);
  });
});


// ── UI helpers ───────────────────────────────────────────────────────────
function setLoading(on, msg) {
  document.getElementById('loading').style.display = on ? 'block' : 'none';
  if (msg) document.getElementById('loading-msg').textContent = msg;
  const btn = document.getElementById('calibrate-btn');
  if (btn) btn.disabled = on || !(_tapePoints.length >= 2 && _mpPoint);
  const abtn = document.getElementById('analyze-btn');
  if (abtn) abtn.disabled = on;
}
function showError(msg) {
  const el = document.getElementById('error-notice');
  el.textContent = msg; el.style.display = 'block';
}
function clearError() { document.getElementById('error-notice').style.display = 'none'; }
