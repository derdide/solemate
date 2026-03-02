/* app.js — Ski Binding Conflict Detector frontend */
'use strict';

// ── Global state ─────────────────────────────────────────────────────────
let _token       = sessionStorage.getItem('sbc_token') || '';
let _imageFile   = null;
let _imageEl     = null;   // Image() object for canvas drawing
let _lastResp    = null;   // last /api/analyze response

// Calibration state
// ALL click coordinates are stored in CANVAS NATURAL PIXELS = image pixels.
// _calScale = displayWidth / naturalWidth — used ONLY to size visual elements
// (dots, line widths) so they appear at a fixed display size regardless of zoom.
let _calMode      = null;   // 'tape' | 'mp' | 'trans1' | 'trans2' | null
let _tapeP1       = null;   // {x, y} in canvas natural pixels (= image pixels)
let _tapeP2       = null;
let _mpPoint      = null;
let _transL1      = null;   // {a:{x,y}, b:{x,y}} — edge-to-edge line, heel side
let _transL2      = null;   // {a:{x,y}, b:{x,y}} — edge-to-edge line, tip side
let _transL1Start = null;   // partial: first click waiting for second
let _transL2Start = null;
let _axisFlipped  = false;  // user toggled "↩ Flip axis" to reverse computed direction
let _calScale     = 1.0;    // displayW / naturalWidth

// Manual hole editing
let _editMode    = false;
let _editedHoles = null;   // null = use auto-detected; array = user-edited list

// ── Token gate ───────────────────────────────────────────────────────────
(function initTokenGate() {
  if (_token) {
    fetch(`/api/health?token=${encodeURIComponent(_token)}`)
      .then(r => { if (r.ok) dismissGate(); })
      .catch(() => {});
  }
})();

function submitToken() {
  const val = document.getElementById('token-input').value.trim();
  fetch(`/api/health?token=${encodeURIComponent(val)}`)
    .then(r => {
      if (r.ok) { _token = val; sessionStorage.setItem('sbc_token', val); dismissGate(); }
      else       { document.getElementById('token-error').style.display = 'block'; }
    })
    .catch(() => { document.getElementById('token-error').style.display = 'block'; });
}
function dismissGate() { document.getElementById('token-gate').style.display = 'none'; }
document.getElementById('token-input').addEventListener('keydown', e => { if (e.key === 'Enter') submitToken(); });

// ── File pick / drag-drop ────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
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

// ── Main analysis ────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!_imageFile) { showError('Please select a ski photo first.'); return; }
  const bsl = parseFloat(document.getElementById('bsl-input').value);
  if (isNaN(bsl) || bsl < 200 || bsl > 420) { showError('Enter a valid BSL (200–420 mm).'); return; }

  clearError();
  // Reset edit mode, transversal calibration, and flip for fresh auto-detect run
  _editMode = false;
  _editedHoles = null;
  _transL1 = null; _transL2 = null; _transL1Start = null; _transL2Start = null;
  _axisFlipped = false;
  const btnFlip = document.getElementById('btn-flip');
  if (btnFlip) btnFlip.classList.remove('active');
  updateEditButton();

  setLoading(true, 'Detecting holes and checking bindings… (may take 10–30s if using AI)');

  const fd = new FormData();
  fd.append('image', _imageFile);
  fd.append('bsl_mm', bsl);
  fd.append('category', document.getElementById('cat-select').value);
  fd.append('min_separation_mm', document.getElementById('sep-input').value);
  fd.append('top_n', 999);
  fd.append('bsl_test_step', 0);
  fd.append('bsl_test_range', 0);

  try {
    const resp = await fetch(`/api/analyze?token=${encodeURIComponent(_token)}`, { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(`Server error ${resp.status}: ${await resp.text()}`);
    _lastResp = await resp.json();
    renderResults(_lastResp);
  } catch (err) {
    showError(err.message || 'Unknown error.');
  } finally {
    setLoading(false);
  }
}

// ── Calibration with manual override ────────────────────────────────────
async function applyCalibrationAndAnalyze() {
  if (!_imageFile) { showError('No image loaded.'); return; }
  if (!_tapeP1 || !_tapeP2) { showError('Click two tape reference points first.'); return; }
  if (!_mpPoint)             { showError('Click the mounting point on the ski.'); return; }

  const refMm = parseFloat(document.getElementById('ref-mm').value);
  if (isNaN(refMm) || refMm <= 0) { showError('Enter a valid reference distance.'); return; }

  // _tapeP1/_tapeP2/_mpPoint are in canvas natural pixels = image pixels (1:1)
  const distPx     = Math.hypot(_tapeP2.x - _tapeP1.x, _tapeP2.y - _tapeP1.y);
  const mmPerPixel = refMm / distPx;

  // Ski axis: heel→tip unit vector in image space.
  // Best source: ③/④ transversal edge-to-edge lines (L1=heel side, L2=tip side)
  //   → midpoints A (heel) and B (tip) → axis = normalize(B−A)
  //   → mounting point is projected onto this centerline for accuracy.
  // Fallback: tape P1→P2 direction (P1=heel, P2=tip per on-screen instruction → heel→tip).
  // User can correct with "↩ Flip axis" button if the orange arrow points the wrong way.
  let axisDx, axisDy;
  let mpX = _mpPoint.x, mpY = _mpPoint.y;

  if (_transL1 && _transL2) {
    const midA = { x: (_transL1.a.x + _transL1.b.x) / 2, y: (_transL1.a.y + _transL1.b.y) / 2 };
    const midB = { x: (_transL2.a.x + _transL2.b.x) / 2, y: (_transL2.a.y + _transL2.b.y) / 2 };
    const dx = midB.x - midA.x, dy = midB.y - midA.y;
    const len = Math.hypot(dx, dy) || 1;
    axisDx = dx / len;
    axisDy = dy / len;
    // Project mounting point onto the ski centerline (removes lateral error)
    const t = (_mpPoint.x - midA.x) * axisDx + (_mpPoint.y - midA.y) * axisDy;
    mpX = midA.x + t * axisDx;
    mpY = midA.y + t * axisDy;
  } else {
    // P1→P2 direction. Instruction says: P1 = heel side, P2 = tip side.
    // So P1→P2 = heel→tip. Use directly (do NOT negate).
    const axLen = distPx || 1;
    axisDx = (_tapeP2.x - _tapeP1.x) / axLen;
    axisDy = (_tapeP2.y - _tapeP1.y) / axLen;
  }

  // Apply user flip toggle (if arrow was pointing wrong way)
  if (_axisFlipped) { axisDx = -axisDx; axisDy = -axisDy; }

  clearError();
  setLoading(true, 'Re-analyzing with manual calibration…');

  const fd = new FormData();
  fd.append('image', _imageFile);
  fd.append('bsl_mm', document.getElementById('bsl-input').value);
  fd.append('category', document.getElementById('cat-select').value);
  fd.append('min_separation_mm', document.getElementById('sep-input').value);
  fd.append('top_n', 999);
  fd.append('bsl_test_step', 0);
  fd.append('bsl_test_range', 0);
  fd.append('mm_per_pixel', mmPerPixel);
  fd.append('mounting_point_x_px', mpX);
  fd.append('mounting_point_y_px', mpY);
  fd.append('axis_dx', axisDx);
  fd.append('axis_dy', axisDy);

  // Always send the current hole list so the server never re-runs auto-detection.
  // Use manually edited holes if available, otherwise use the last auto-detected set (or empty).
  const currentHoles = _editedHoles !== null
    ? _editedHoles
    : (_lastResp?.holes_px || []);
  fd.append('override_holes_json', JSON.stringify(currentHoles));

  try {
    const resp = await fetch(`/api/analyze?token=${encodeURIComponent(_token)}`, { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(`Server error ${resp.status}: ${await resp.text()}`);
    _lastResp = await resp.json();
    renderResults(_lastResp);
  } catch (err) {
    showError(err.message || 'Unknown error.');
  } finally {
    setLoading(false);
  }
}

// ── Calibration canvas ───────────────────────────────────────────────────
const calCanvas = document.getElementById('cal-canvas');
const calCtx    = calCanvas.getContext('2d');

function drawCalCanvas() {
  if (!_imageEl) return;
  const displayW = calCanvas.offsetWidth || 340;
  const ratio = _imageEl.naturalHeight / _imageEl.naturalWidth;
  calCanvas.width  = _imageEl.naturalWidth;
  calCanvas.height = _imageEl.naturalHeight;
  calCanvas.style.height = (displayW * ratio) + 'px';

  // _calScale: display pixels per canvas pixel — used for sizing only
  _calScale = displayW / _imageEl.naturalWidth;

  calCtx.drawImage(_imageEl, 0, 0);

  // Draw holes (edited list or auto-detected from last response)
  const holeList = _editedHoles !== null ? _editedHoles : (_lastResp?.holes_px || []);
  holeList.forEach(h => {
    const r = Math.max(h.radius_px || 5, 15 / _calScale);
    calCtx.beginPath();
    calCtx.arc(h.x_px, h.y_px, r, 0, Math.PI * 2);
    calCtx.strokeStyle = h.source === 'manual' ? 'rgba(50,255,100,0.9)' : 'rgba(255,200,0,0.9)';
    calCtx.lineWidth = 3 / _calScale;
    calCtx.stroke();
  });

  // Draw tape reference points — stored in canvas natural pixels, use directly
  if (_tapeP1) drawDot(_tapeP1.x, _tapeP1.y, '#00e5ff', '1');
  if (_tapeP2) {
    drawDot(_tapeP2.x, _tapeP2.y, '#00e5ff', '2');
    if (_tapeP1) {
      calCtx.beginPath();
      calCtx.moveTo(_tapeP1.x, _tapeP1.y);
      calCtx.lineTo(_tapeP2.x, _tapeP2.y);
      calCtx.strokeStyle = '#00e5ff';
      calCtx.lineWidth = 2 / _calScale;
      calCtx.setLineDash([8 / _calScale, 4 / _calScale]);
      calCtx.stroke();
      calCtx.setLineDash([]);
      const refMm = parseFloat(document.getElementById('ref-mm').value) || 100;
      // Distance in canvas natural pixels = image pixels, so mpp is accurate
      const distPx = Math.hypot(_tapeP2.x - _tapeP1.x, _tapeP2.y - _tapeP1.y);
      const mpp = refMm / distPx;
      document.getElementById('cal-status').textContent =
        `Tape: ${distPx.toFixed(0)}px = ${refMm}mm → scale = ${mpp.toFixed(4)} mm/px`;
    }
  }

  // Draw mounting point crosshair — stored in canvas natural pixels, use directly
  if (_mpPoint) {
    const arm = 16 / _calScale;
    calCtx.strokeStyle = '#ffd700';
    calCtx.lineWidth   = 3 / _calScale;
    calCtx.beginPath(); calCtx.moveTo(_mpPoint.x - arm, _mpPoint.y); calCtx.lineTo(_mpPoint.x + arm, _mpPoint.y); calCtx.stroke();
    calCtx.beginPath(); calCtx.moveTo(_mpPoint.x, _mpPoint.y - arm); calCtx.lineTo(_mpPoint.x, _mpPoint.y + arm); calCtx.stroke();
  }

  // Draw orange "→ TIP" arrow showing the computed ski axis direction.
  // This lets the user verify before clicking Analyze — flip with the "↩ Flip axis" button if wrong.
  if (_mpPoint && ((_tapeP1 && _tapeP2) || (_transL1 && _transL2))) {
    let adx, ady;
    if (_transL1 && _transL2) {
      const midA = { x: (_transL1.a.x + _transL1.b.x) / 2, y: (_transL1.a.y + _transL1.b.y) / 2 };
      const midB = { x: (_transL2.a.x + _transL2.b.x) / 2, y: (_transL2.a.y + _transL2.b.y) / 2 };
      const dx = midB.x - midA.x, dy = midB.y - midA.y;
      const lenAB = Math.hypot(dx, dy) || 1;
      adx = dx / lenAB; ady = dy / lenAB;
    } else {
      const dx = _tapeP2.x - _tapeP1.x, dy = _tapeP2.y - _tapeP1.y;
      const lenT = Math.hypot(dx, dy) || 1;
      adx = dx / lenT; ady = dy / lenT;   // P1→P2 = heel→tip (per instruction)
    }
    if (_axisFlipped) { adx = -adx; ady = -ady; }
    drawAxisArrow(_mpPoint.x, _mpPoint.y, adx, ady);
  }

  // Draw transversal edge-to-edge lines (③ L1 / ④ L2) and derived centerline
  if (_transL1)       drawTransLine(_transL1, '#ff55ff', 'L1');
  if (_transL1Start)  drawDot(_transL1Start.x, _transL1Start.y, '#ff55ff', 'L1…');
  if (_transL2)       drawTransLine(_transL2, '#cc44cc', 'L2');
  if (_transL2Start)  drawDot(_transL2Start.x, _transL2Start.y, '#cc44cc', 'L2…');

  if (_transL1 && _transL2) {
    const midA = { x: (_transL1.a.x + _transL1.b.x) / 2, y: (_transL1.a.y + _transL1.b.y) / 2 };
    const midB = { x: (_transL2.a.x + _transL2.b.x) / 2, y: (_transL2.a.y + _transL2.b.y) / 2 };
    const dx = midB.x - midA.x, dy = midB.y - midA.y;
    const lenAB = Math.hypot(dx, dy) || 1;
    const ndx = dx / lenAB, ndy = dy / lenAB;
    const ext = Math.max(calCanvas.width, calCanvas.height);
    calCtx.beginPath();
    calCtx.moveTo(midA.x - ndx * ext, midA.y - ndy * ext);
    calCtx.lineTo(midA.x + ndx * ext, midA.y + ndy * ext);
    calCtx.strokeStyle = 'rgba(255,100,255,0.65)';
    calCtx.lineWidth = 2 / _calScale;
    calCtx.setLineDash([8 / _calScale, 4 / _calScale]);
    calCtx.stroke();
    calCtx.setLineDash([]);
    drawDot(midA.x, midA.y, '#ff55ff', 'A');
    drawDot(midB.x, midB.y, '#ff55ff', 'B');
  }

  // Edit mode overlay label
  if (_editMode) {
    const fs = 14 / _calScale;
    calCtx.font = `bold ${fs}px sans-serif`;
    calCtx.fillStyle = 'rgba(50,255,100,0.95)';
    calCtx.fillText('✏ EDIT — click=add hole, click on hole=remove', 8 / _calScale, fs + 4 / _calScale);
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
  // Draw the edge-to-edge line
  calCtx.beginPath();
  calCtx.moveTo(line.a.x, line.a.y);
  calCtx.lineTo(line.b.x, line.b.y);
  calCtx.strokeStyle = color;
  calCtx.lineWidth = 2 / _calScale;
  calCtx.stroke();
  // Endpoints (unlabelled)
  drawDot(line.a.x, line.a.y, color, '');
  drawDot(line.b.x, line.b.y, color, '');
  // Midpoint with label
  const mx = (line.a.x + line.b.x) / 2, my = (line.a.y + line.b.y) / 2;
  drawDot(mx, my, color, label);
}

// Draw an orange "→ TIP" arrow from (ox, oy) in direction (adx, ady)
function drawAxisArrow(ox, oy, adx, ady) {
  const len = 100 / _calScale;
  const hw  =  12 / _calScale;
  const ex  = ox + adx * len, ey = oy + ady * len;
  calCtx.strokeStyle = '#ff6600';
  calCtx.fillStyle   = '#ff6600';
  calCtx.lineWidth   = 3 / _calScale;
  // Shaft
  calCtx.beginPath(); calCtx.moveTo(ox, oy); calCtx.lineTo(ex, ey); calCtx.stroke();
  // Arrowhead
  calCtx.beginPath();
  calCtx.moveTo(ex, ey);
  calCtx.lineTo(ex - adx*hw*1.5 - ady*hw, ey - ady*hw*1.5 + adx*hw);
  calCtx.lineTo(ex - adx*hw*1.5 + ady*hw, ey - ady*hw*1.5 - adx*hw);
  calCtx.closePath(); calCtx.fill();
  // Label
  calCtx.font = `bold ${12/_calScale}px sans-serif`;
  calCtx.fillStyle = '#ff6600';
  calCtx.fillText('→TIP', ex + adx*(hw+2/_calScale), ey + ady*(hw+2/_calScale));
}

// Toggle ski axis direction — re-check overlay after clicking if wrong
function flipAxis() {
  _axisFlipped = !_axisFlipped;
  const btn = document.getElementById('btn-flip');
  if (btn) btn.classList.toggle('active', _axisFlipped);
  drawCalCanvas();
}

// Canvas click handler — converts to canvas natural pixels (zoom-proof)
calCanvas.addEventListener('click', e => {
  const rect = calCanvas.getBoundingClientRect();
  // Compute position in canvas natural pixel space (= image pixels).
  // Using rect.width (from getBoundingClientRect) rather than offsetWidth correctly
  // accounts for CSS transforms and browser zoom.
  const natX = (e.clientX - rect.left) * (calCanvas.width  / (rect.width  || 1));
  const natY = (e.clientY - rect.top)  * (calCanvas.height / (rect.height || 1));

  if (_editMode) {
    _handleEditClick(natX, natY);
    drawCalCanvas();
    return;
  }

  if (!_calMode) return;

  if (_calMode === 'tape') {
    if (!_tapeP1) {
      _tapeP1 = { x: natX, y: natY };
      document.getElementById('cal-mode-label').textContent = 'Now click the SECOND tape mark — tip side of ski.';
    } else {
      _tapeP2 = { x: natX, y: natY };
      _calMode = null;
      updateCalButtons();
      document.getElementById('cal-mode-label').textContent = '✓ Two tape points set. Now click "② Click mounting pt".';
    }
  } else if (_calMode === 'mp') {
    _mpPoint = { x: natX, y: natY };
    _calMode = null;
    updateCalButtons();
    document.getElementById('cal-mode-label').textContent = '✓ Mounting point set. Now click "③ Axis L1" (heel side) to set ski axis — or click "Analyze →" if already set.';
  } else if (_calMode === 'trans1') {
    if (!_transL1Start) {
      _transL1Start = { x: natX, y: natY };
      document.getElementById('cal-mode-label').textContent = 'L1: Now click the OPPOSITE ski edge.';
    } else {
      _transL1 = { a: _transL1Start, b: { x: natX, y: natY } };
      _transL1Start = null;
      _calMode = null;
      updateCalButtons();
      document.getElementById('cal-mode-label').textContent = '✓ L1 set. Now click "④ Axis L2" (tip side of ski).';
    }
  } else if (_calMode === 'trans2') {
    if (!_transL2Start) {
      _transL2Start = { x: natX, y: natY };
      document.getElementById('cal-mode-label').textContent = 'L2: Now click the OPPOSITE ski edge.';
    } else {
      _transL2 = { a: _transL2Start, b: { x: natX, y: natY } };
      _transL2Start = null;
      _calMode = null;
      updateCalButtons();
      document.getElementById('cal-mode-label').textContent = '✓ Centerline set from L1 & L2. Click "Analyze →".';
    }
  }
  drawCalCanvas();
});

function startCalMode(mode) {
  _calMode = mode;
  updateCalButtons();
  if (mode === 'tape') {
    _tapeP1 = _tapeP2 = null;
    _axisFlipped = false;
    const btnFlip2 = document.getElementById('btn-flip');
    if (btnFlip2) btnFlip2.classList.remove('active');
    document.getElementById('cal-mode-label').textContent = 'Click the FIRST tape mark — HEEL side (P1). Then click tip side (P2).';
  } else if (mode === 'mp') {
    document.getElementById('cal-mode-label').textContent = 'Click the ski CENTRE MARK (mounting point / half-sole line on ski).';
  } else if (mode === 'trans1') {
    _transL1 = null; _transL1Start = null;
    document.getElementById('cal-mode-label').textContent = 'L1 (heel side): Click one ski EDGE — then click the OPPOSITE edge to complete the line.';
  } else if (mode === 'trans2') {
    _transL2 = null; _transL2Start = null;
    document.getElementById('cal-mode-label').textContent = 'L2 (tip side): Click one ski EDGE — then click the OPPOSITE edge to complete the line.';
  }
  drawCalCanvas();
}

function updateCalButtons() {
  document.getElementById('btn-tape').classList.toggle('active', _calMode === 'tape');
  document.getElementById('btn-mp').classList.toggle('active', _calMode === 'mp');
  const btnTrans1 = document.getElementById('btn-trans1');
  if (btnTrans1) btnTrans1.classList.toggle('active', _calMode === 'trans1');
  const btnTrans2 = document.getElementById('btn-trans2');
  if (btnTrans2) btnTrans2.classList.toggle('active', _calMode === 'trans2');
}

// Redraw on window resize — recalculates _calScale for the new display size
window.addEventListener('resize', () => { if (_imageEl) drawCalCanvas(); });

// ── Manual hole editing ───────────────────────────────────────────────────
function toggleEditMode() {
  _editMode = !_editMode;
  // Initialise edit list from auto-detected holes on first activation
  if (_editMode && _editedHoles === null && _lastResp) {
    _editedHoles = (_lastResp.holes_px || []).map(h => ({ ...h }));
  }
  updateEditButton();
  const count = (_editedHoles || []).length;
  const btnClear = document.getElementById('btn-clear-holes');
  if (_editMode) {
    document.getElementById('cal-mode-label').textContent =
      `✏ EDIT MODE — ${count} hole(s). Click the photo to add a hole; click an existing circle to remove it.`;
    // Highlight canvas with a green border so the user sees where to click
    calCanvas.style.outline = '3px solid rgba(50,255,100,0.85)';
    if (btnClear) btnClear.style.display = '';
  } else {
    document.getElementById('cal-mode-label').textContent = '';
    calCanvas.style.outline = '';
    if (btnClear) btnClear.style.display = 'none';
  }
  drawCalCanvas();
}

function clearAllHoles() {
  _editedHoles = [];
  const count = 0;
  document.getElementById('cal-mode-label').textContent =
    `✏ EDIT MODE — 0 hole(s). Click on image to add holes.`;
  drawCalCanvas();
}

function updateEditButton() {
  const btn = document.getElementById('btn-edit');
  if (btn) btn.classList.toggle('active', _editMode);
}

function _handleEditClick(natX, natY) {
  if (_editedHoles === null) {
    _editedHoles = (_lastResp?.holes_px || []).map(h => ({ ...h }));
  }
  // Click threshold ≈ 20 display pixels, expressed in canvas natural pixel space
  const threshold = 20 / _calScale;
  const idx = _editedHoles.findIndex(h =>
    Math.hypot(h.x_px - natX, h.y_px - natY) < threshold
  );
  if (idx >= 0) {
    _editedHoles.splice(idx, 1);
    document.getElementById('cal-mode-label').textContent =
      `Removed — ${_editedHoles.length} hole(s) remaining.`;
  } else {
    const defaultRadius = _lastResp?.holes_px?.[0]?.radius_px || 5;
    _editedHoles.push({ x_px: Math.round(natX), y_px: Math.round(natY), radius_px: defaultRadius, confidence: 1.0, source: 'manual' });
    document.getElementById('cal-mode-label').textContent =
      `Added — ${_editedHoles.length} hole(s) total.`;
  }
}

// ── Render results ───────────────────────────────────────────────────────
function renderResults(data) {
  // Always redraw canvas (step 1 shows detected/edited holes)
  drawCalCanvas();

  // Switch to step 2 only when we have a fully calibrated result
  if (data.mounting_point_known) {
    showStep(2);
  }
  // else: stay on step 1 so the user can see detected holes and then calibrate

  // Calibration status
  const calOk     = document.getElementById('cal-ok');
  const calNotice = document.getElementById('cal-notice');
  if (data.calibration_available && data.mounting_point_known) {
    calOk.style.display = 'block'; calNotice.style.display = 'none';
    calOk.textContent = `✓ Scale: ${data.calibration.mm_per_pixel.toFixed(4)} mm/px. Mounting point set.`;
  } else if (data.calibration_available && !data.mounting_point_known) {
    calOk.style.display = 'none'; calNotice.style.display = 'block';
    calNotice.textContent = '⚠ Scale detected but mounting point unknown — click "② Click mounting pt" to set it.';
  } else {
    calOk.style.display = 'none'; calNotice.style.display = 'block';
  }

  // Hole count caption
  document.getElementById('hole-count').textContent =
    `Detected ${data.detected_holes} hole(s)` +
    (data.calibration_available
      ? ` · scale ${data.calibration.mm_per_pixel.toFixed(4)} mm/px`
      : ' · no scale') +
    (data.mounting_point_known ? ' · mounting point set' : ' · mounting point unknown');

  // Annotated overlay image (shown when full calibration available)
  const overlayImg = document.getElementById('overlay-img');
  const previewCvs = document.getElementById('preview-canvas');
  if (data.output_image_base64) {
    overlayImg.src = 'data:image/jpeg;base64,' + data.output_image_base64;
    overlayImg.style.display = 'block';
    previewCvs.style.display = 'none';
  } else {
    overlayImg.style.display = 'none';
    drawPreviewCanvas(data.holes_px || []);
    previewCvs.style.display = 'block';
  }

  // Identified binding banner
  const idBanner = document.getElementById('id-banner');
  if (data.identify_matches && data.identify_matches.length > 0 && data.identify_matches[0].match_score > 0.5) {
    const top = data.identify_matches[0];
    idBanner.style.display = 'block';
    idBanner.innerHTML =
      `🔍 Previously mounted: <strong>${top.binding_name}</strong> ` +
      `at BSL ${top.bsl_mm}mm ` +
      `(${Math.round(top.match_score * 100)}% match, ${top.holes_matched}/${top.holes_total} holes)` +
      (top.verified ? '' : ' <em>⚠ unverified template</em>');
  } else {
    idBanner.style.display = 'none';
  }

  // Results table
  const tableCard = document.getElementById('results-table-card');
  const noCal     = document.getElementById('no-cal-hint');
  const tbody     = document.getElementById('results-body');
  tbody.innerHTML = '';

  if (!data.mounting_point_known) {
    tableCard.style.display = 'none';
    noCal.style.display = 'block';
    return;
  }

  noCal.style.display = 'none';
  tableCard.style.display = 'block';

  if (!data.results || data.results.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="7" style="text-align:center;color:var(--text-dim)">No compatible bindings found for this BSL.</td>';
    tbody.appendChild(tr);
    return;
  }

  document.getElementById('img-binding-label').textContent =
    data.results.length ? `— ${data.results[0].binding_name}` : '';

  // BSL the user typed in — used to flag rows where a different size fits better
  const enteredBsl = parseFloat(document.getElementById('bsl-input').value) || 0;

  data.results.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.className = r.is_mountable ? 'mountable' : '';
    if (idx === 0) tr.classList.add('selected');

    const heelOff = r.heel_offset_mm !== 0
      ? `${r.heel_offset_mm > 0 ? '+' : ''}${r.heel_offset_mm.toFixed(1)}mm` : '—';
    const statusBadge = r.is_mountable
      ? `<span class="badge badge-ok">OK</span>`
      : `<span class="badge badge-err">CONFLICT</span>`;
    const unverified  = r.verified ? '' : ' <span class="badge badge-warn" title="Unverified template">⚠</span>';
    const varLabel    = r.variant_id ? ` <small style="color:var(--text-dim)">[${r.variant_id}]</small>` : '';

    // Highlight BSL when it differs from the user-entered value
    const bslDiff   = r.bsl_mm - enteredBsl;
    const bslStyle  = bslDiff !== 0
      ? 'font-weight:bold;color:var(--accent)'
      : 'color:var(--text-dim)';
    const bslLabel  = bslDiff === 0 ? `${r.bsl_mm}`
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
}

// ── Preview canvas (holes only, no calibration) ──────────────────────────
function drawPreviewCanvas(holesPx) {
  const cvs = document.getElementById('preview-canvas');
  const ctx = cvs.getContext('2d');
  if (!_imageEl) return;

  cvs.width  = _imageEl.naturalWidth;
  cvs.height = _imageEl.naturalHeight;
  const displayW = cvs.offsetWidth || 600;
  cvs.style.height = (displayW * _imageEl.naturalHeight / _imageEl.naturalWidth) + 'px';

  ctx.drawImage(_imageEl, 0, 0);
  holesPx.forEach(h => {
    ctx.beginPath();
    ctx.arc(h.x_px, h.y_px, Math.max(h.radius_px, 8), 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 220, 0, 0.9)';
    ctx.lineWidth   = 3;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(h.x_px, h.y_px, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,220,0,0.8)';
    ctx.fill();
  });
}

// ── Select a binding row → re-visualize ─────────────────────────────────
async function selectBinding(tr, result) {
  document.querySelectorAll('#results-body tr').forEach(r => r.classList.remove('selected'));
  tr.classList.add('selected');
  document.getElementById('img-binding-label').textContent = `— ${result.binding_name}`;
  if (!_lastResp || !_lastResp.mounting_point_known) return;

  const cal = _lastResp.calibration;
  // Axis unit vector from calibration (might be from manual or auto)
  const adx = cal.axis_dx ?? -1.0;
  const ady = cal.axis_dy ??  0.0;
  // pixels_to_ski_mm — orientation-aware (matches Python binding_matcher.pixels_to_ski_mm)
  function pxToSki(h) {
    const vx = h.x_px - cal.mounting_point_x_px;
    const vy = h.y_px - cal.ski_centerline_y_px;
    return {
      x_abs:  (adx * vx + ady * vy) * cal.mm_per_pixel,
      y_abs:  (ady * vx - adx * vy) * cal.mm_per_pixel,
    };
  }

  const fd  = new FormData();
  fd.append('image', _imageFile);
  fd.append('binding_id', result.binding_id);
  fd.append('bsl_mm', result.bsl_mm);
  fd.append('variant_id', result.variant_id || '');
  fd.append('min_separation_mm', document.getElementById('sep-input').value);
  fd.append('existing_holes_json', JSON.stringify(
    (_lastResp.holes_px || []).map(h => pxToSki(h))
  ));
  fd.append('mm_per_pixel', cal.mm_per_pixel);
  fd.append('mounting_point_x_px', cal.mounting_point_x_px);
  fd.append('mounting_point_y_px', cal.ski_centerline_y_px);
  fd.append('axis_dx', adx);
  fd.append('axis_dy', ady);

  try {
    const resp = await fetch(`/api/visualize?token=${encodeURIComponent(_token)}`, { method: 'POST', body: fd });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.output_image_base64) {
      document.getElementById('overlay-img').src = 'data:image/jpeg;base64,' + data.output_image_base64;
      document.getElementById('overlay-img').style.display = 'block';
      document.getElementById('preview-canvas').style.display = 'none';
    }
  } catch (_) {}
}

// ── Step navigation ──────────────────────────────────────────────────────
function showStep(n) {
  document.getElementById('step1').style.display = n === 1 ? '' : 'none';
  document.getElementById('step2').style.display = n === 2 ? '' : 'none';
  // When returning to step 1, turn off edit mode highlight (but keep _editedHoles)
  if (n === 1) calCanvas.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── UI helpers ───────────────────────────────────────────────────────────
function setLoading(on, msg) {
  document.getElementById('loading').style.display = on ? 'block' : 'none';
  if (msg) document.getElementById('loading-msg').textContent = msg;
  ['analyze-btn', 'detect-btn'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = on;
  });
}
function showError(msg) {
  const el = document.getElementById('error-notice');
  el.textContent = msg; el.style.display = 'block';
}
function clearError() { document.getElementById('error-notice').style.display = 'none'; }
