/* app.js — Ski Binding Conflict Detector UI.
 *
 * Everything runs in this page: the photo is drawn on canvases and never leaves the
 * device. Geometry lives in calib.js (photo ↔ ski mm) and matcher.js (templates vs.
 * holes); print.js builds the 1:1 check sheets.
 */
'use strict';

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Working copies larger than this are scaled down on load: beyond it mobile browsers
// refuse to allocate the canvas, and the extra pixels add nothing to a click's accuracy.
const MAX_WORK_PIXELS = 16e6;
const HOLE_RADIUS_MM = 2.05;          // 4.1 mm drill
const TOUCH_OFFSET_PX = 55;           // on touch, the mark lands this far above the finger

// ── State ────────────────────────────────────────────────────────────────
let _db = null;                       // binding_db.json
let _img = null, _imgW = 0, _imgH = 0;

// Step 1 — calibration clicks, in image pixels
let _calMode      = null;             // 'tape' | 'mp' | 'trans1' | 'trans2' | null
let _tapePoints   = [];
let _mpPoint      = null;
let _transL1      = null, _transL2      = null;   // {a:{x,y}, b:{x,y}} edge-to-edge lines
let _transL1Start = null, _transL2Start = null;
let _axisFlipped  = false;

// Step 2
let _cal   = null;                    // Calib calibration
let _holes = [];                      // existing holes, image pixels

// Step 3
let _existing   = [];                 // existing holes, ski mm
let _results    = [];
let _sel        = null;               // { binding, heelAuto }
let _view       = null;               // analysis of the selected binding at the slider values
let _clearCache = null;               // { key, result } for the mount-offset suggestion
let _identified = null;               // Matcher.identify() output
let _identShown = null;               // match whose pattern is drawn on the photo

let _gridEnabled = false, _gridSpacingMm = 10;

function esc(value) {
  return String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const signed = (n, digits = 1) => (n > 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n).toFixed(digits);


// ── Zoomable, pannable canvas panel ──────────────────────────────────────
// Mouse: click = tap, drag = pan, wheel = zoom.
// Touch: two fingers pinch-zoom and pan. One finger pans, except on panels that place
// marks: there it drags a preview that sits above the fingertip and drops on release.
class ZoomPanel {
  constructor(name, opts) {
    this.panel  = $(name + '-panel');
    this.canvas = $(name + '-canvas');
    this.label  = $(name + '-zoom-label');
    this.ctx    = this.canvas.getContext('2d');
    this.opts   = opts;
    this.zoom   = 1;
    this.ptrs   = new Map();
    this.tap = null; this.pinch = null; this.preview = null; this.raf = 0;

    const c = this.canvas;
    c.addEventListener('pointerdown',   e => this._down(e));
    c.addEventListener('pointermove',   e => this._move(e));
    c.addEventListener('pointerup',     e => this._up(e, true));
    c.addEventListener('pointercancel', e => this._up(e, false));
    this.panel.addEventListener('wheel', e => {
      if (!_img) return;
      e.preventDefault();
      const r = this.panel.getBoundingClientRect();
      this.setZoom(this.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), { x: e.clientX - r.left, y: e.clientY - r.top });
    }, { passive: false });
  }

  /** Screen pixels per image pixel. */
  get scale() { return (this.canvas.getBoundingClientRect().width || 1) / (this.canvas.width || 1); }

  setImageSize(w, h) { this.canvas.width = w; this.canvas.height = h; this.setZoom(1); }

  redraw() { if (_img && this.panel.offsetParent !== null) this.opts.draw(this.ctx, 1 / this.scale, this); }
  requestRedraw() { if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.redraw(); }); }

  _applyZoom(z) {
    this.zoom = clamp(z, 1, 8);
    this.canvas.style.width = this.zoom > 1 ? (this.zoom * 100) + '%' : '';
    if (this.label) this.label.textContent = Math.round(this.zoom * 100) + '%';
  }

  /** Zoom keeping the image point under `focal` (panel coordinates) where it is. */
  setZoom(level, focal) {
    const old = this.zoom;
    const fx = focal ? focal.x : this.panel.clientWidth / 2;
    const fy = focal ? focal.y : this.panel.clientHeight / 2;
    const ix = (this.panel.scrollLeft + fx) / old, iy = (this.panel.scrollTop + fy) / old;
    this._applyZoom(level);
    this.panel.scrollLeft = ix * this.zoom - fx;
    this.panel.scrollTop  = iy * this.zoom - fy;
    this.redraw();
  }

  /** Does a one-finger drag position a mark (rather than pan) right now? */
  _places(e) {
    return e.pointerType !== 'mouse' && !!this.opts.onTap && (!this.opts.canPlace || this.opts.canPlace());
  }

  _nat(e) {
    const r = this.canvas.getBoundingClientRect();
    const k = this.canvas.width / (r.width || 1);
    const lift = this._places(e) ? TOUCH_OFFSET_PX : 0;
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top - lift) * k };
  }

  _mid() {
    const [a, b] = [...this.ptrs.values()];
    const r = this.panel.getBoundingClientRect();
    return { x: (a.x + b.x) / 2 - r.left, y: (a.y + b.y) / 2 - r.top, d: Math.hypot(b.x - a.x, b.y - a.y) || 1 };
  }

  _down(e) {
    if (!_img) return;
    try { this.canvas.setPointerCapture(e.pointerId); } catch (_) { /* not capturable */ }
    this.ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.ptrs.size === 1) {
      this.tap = { x: e.clientX, y: e.clientY, sl: this.panel.scrollLeft, st: this.panel.scrollTop, moved: false };
      if (this._places(e)) { this.preview = this._nat(e); this.requestRedraw(); }
    } else if (this.ptrs.size === 2) {
      this.tap = null; this.preview = null;
      const m = this._mid();
      this.pinch = { d0: m.d, z0: this.zoom, ix: (this.panel.scrollLeft + m.x) / this.zoom, iy: (this.panel.scrollTop + m.y) / this.zoom };
      this.requestRedraw();
    }
  }

  _move(e) {
    if (!this.ptrs.has(e.pointerId)) return;
    this.ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.ptrs.size >= 2 && this.pinch) {
      const m = this._mid();
      this._applyZoom(this.pinch.z0 * m.d / this.pinch.d0);
      this.panel.scrollLeft = this.pinch.ix * this.zoom - m.x;
      this.panel.scrollTop  = this.pinch.iy * this.zoom - m.y;
      this.requestRedraw();
      return;
    }
    if (this.ptrs.size === 1 && this.tap) {
      const dx = e.clientX - this.tap.x, dy = e.clientY - this.tap.y;
      if (Math.hypot(dx, dy) >= 5) this.tap.moved = true;
      if (this._places(e)) {
        this.preview = this._nat(e);
        this.requestRedraw();
      } else if (this.tap.moved) {
        this.panel.scrollLeft = this.tap.sl - dx;
        this.panel.scrollTop  = this.tap.st - dy;
        this.canvas.style.cursor = 'grabbing';
      }
    }
  }

  _up(e, completed) {
    if (!this.ptrs.has(e.pointerId)) return;
    const tap = this.ptrs.size === 1 ? this.tap : null;
    const places = this._places(e);
    const pos = this._nat(e);
    this.ptrs.delete(e.pointerId);
    if (this.ptrs.size < 2) this.pinch = null;
    if (this.ptrs.size === 0) this.tap = null;
    this.preview = null;
    this.canvas.style.cursor = '';
    const canTap = this.opts.onTap && (!this.opts.canPlace || this.opts.canPlace());
    if (completed && tap && canTap && (places || !tap.moved)) this.opts.onTap(pos, this);
    this.redraw();
  }
}

const calPanel   = new ZoomPanel('cal',   { draw: drawCalCanvas,   onTap: onCalTap,  canPlace: () => !!_calMode });
const holesPanel = new ZoomPanel('holes', { draw: drawHolesCanvas, onTap: onHoleTap, canPlace: () => !!_cal });
const resPanel   = new ZoomPanel('res',   { draw: drawResCanvas });
const PANELS = { cal: calPanel, holes: holesPanel, res: resPanel };

document.querySelectorAll('.zoom-btn').forEach(btn => btn.addEventListener('click', () => {
  const panel = PANELS[btn.dataset.zoom], dir = Number(btn.dataset.dir);
  panel.setZoom(dir === 0 ? 1 : panel.zoom * (dir > 0 ? 1.3 : 1 / 1.3));
}));

window.addEventListener('resize', () => Object.values(PANELS).forEach(p => p.setZoom(1)));


// ── Binding database ─────────────────────────────────────────────────────
function validBinding(b) {
  const holesOk = u => u && Array.isArray(u.holes) && u.holes.length > 0 &&
    u.holes.every(h => Number.isFinite(h.x) && Number.isFinite(h.y));
  return b && typeof b.id === 'string' && typeof b.name === 'string' &&
    ['half_bsl', 'fixed'].includes(b.mounting_reference) && holesOk(b.front_unit) && holesOk(b.heel_unit);
}

fetch('templates/binding_db.json')
  .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
  .then(db => {
    const all = Array.isArray(db.bindings) ? db.bindings : [];
    const good = all.filter(validBinding);
    if (good.length < all.length) console.warn(`${all.length - good.length} binding template(s) skipped: malformed entry`);
    if (!good.length) throw new Error('no usable templates');
    _db = { ...db, bindings: good };
  })
  .catch(err => showError(`Could not load the binding templates (${err.message}). The page must be served over http(s), not opened as a file.`));


// ── Photo ────────────────────────────────────────────────────────────────
const dropZone  = $('drop-zone');
const fileInput = $('file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
});
fileInput.addEventListener('click', e => e.stopPropagation());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  const url = URL.createObjectURL(file);
  const im = new Image();
  im.onerror = () => {
    URL.revokeObjectURL(url);
    showError('This image could not be read. Use a JPEG or PNG (iPhone HEIC photos open in Safari only).');
  };
  im.onload = () => {
    clearError();
    let w = im.naturalWidth, h = im.naturalHeight;
    const k = Math.min(1, Math.sqrt(MAX_WORK_PIXELS / (w * h)));
    if (k < 1) {
      const c = document.createElement('canvas');
      c.width = Math.round(w * k); c.height = Math.round(h * k);
      c.getContext('2d').drawImage(im, 0, 0, c.width, c.height);
      _img = c; w = c.width; h = c.height;
      URL.revokeObjectURL(url);
    } else {
      _img = im;
    }
    _imgW = w; _imgH = h;

    // A new photo invalidates everything measured on the previous one.
    _calMode = null; _tapePoints = []; _mpPoint = null;
    _transL1 = _transL2 = _transL1Start = _transL2Start = null;
    _axisFlipped = false; _cal = null; _holes = [];
    _existing = []; _results = []; _sel = null; _view = null; _identified = null; _identShown = null;
    $('btn-flip').classList.remove('active');
    updateCalButtons(); updateCalibrateButton();
    $('cal-mode-label').textContent = 'Photo loaded — click "① Tape marks" to start.';
    $('cal-status').textContent = '';

    $('drop-label').textContent = file.name;
    dropZone.querySelector('.drop-icon').textContent = '✅';
    Object.values(PANELS).forEach(p => p.setImageSize(w, h));
    showStep(1);
  };
  im.src = url;
}


// ── Shared drawing helpers ───────────────────────────────────────────────
function drawImage(ctx) { ctx.drawImage(_img, 0, 0, _imgW, _imgH); }

function drawCross(ctx, x, y, arm, lw, color, borderColor) {
  ctx.lineCap = 'round';
  const stroke = () => {
    ctx.beginPath(); ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm); ctx.stroke();
  };
  if (borderColor) { ctx.strokeStyle = borderColor; ctx.lineWidth = lw * 2.6; stroke(); }
  ctx.strokeStyle = color; ctx.lineWidth = lw; stroke();
}

function line(ctx, a, b) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }

/** Radius in image px of a real 4.1 mm hole at ski position xMm, never smaller than minScreenPx. */
function holeRadiusPx(xMm, k, minScreenPx) {
  return Math.max(HOLE_RADIUS_MM * Calib.pxPerMm(_cal, xMm), minScreenPx * k);
}

/** Millimetre grid in the ski frame. With parallax correction the cells follow the perspective. */
function drawGrid(ctx, k) {
  const sp = _gridSpacingMm;
  const corners = [[0, 0], [_imgW, 0], [0, _imgH], [_imgW, _imgH]].map(([x, y]) => Calib.toMm(_cal, { x, y }));
  const xMin = Math.min(...corners.map(c => c.x)), xMax = Math.max(...corners.map(c => c.x));
  const yMax = Math.max(...corners.map(c => Math.abs(c.y)));
  if (!Number.isFinite(xMin + xMax + yMax) || (xMax - xMin) / sp > 1500 || yMax / sp > 750) return;
  ctx.save();
  ctx.lineWidth = Math.max(1, k);
  ctx.setLineDash([4 * k, 4 * k]);
  const style = i => { ctx.strokeStyle = i === 0 ? 'rgba(255,255,100,0.55)' : i % 10 === 0 ? 'rgba(255,255,255,0.34)' : 'rgba(255,255,255,0.18)'; };
  for (let i = Math.ceil(xMin / sp); i <= Math.floor(xMax / sp); i++) {
    style(i);
    line(ctx, Calib.toPx(_cal, { x: i * sp, y: -yMax }), Calib.toPx(_cal, { x: i * sp, y: yMax }));
  }
  const n = Math.floor(yMax / sp);
  for (let j = -n; j <= n; j++) {
    style(j);
    line(ctx, Calib.toPx(_cal, { x: xMin, y: j * sp }), Calib.toPx(_cal, { x: xMax, y: j * sp }));
  }
  ctx.restore();
}

document.querySelectorAll('.btn-grid-toggle').forEach(btn => btn.addEventListener('click', () => {
  _gridEnabled = !_gridEnabled;
  document.querySelectorAll('.btn-grid-toggle').forEach(b => {
    b.textContent = _gridEnabled ? 'Grid on' : 'Grid off';
    b.classList.toggle('active', _gridEnabled);
  });
  holesPanel.redraw(); resPanel.redraw();
}));
document.querySelectorAll('.btn-grid-spacing').forEach(btn => btn.addEventListener('click', () => {
  _gridSpacingMm = Number(btn.dataset.mm);
  document.querySelectorAll('.btn-grid-spacing').forEach(b => b.classList.toggle('active', Number(b.dataset.mm) === _gridSpacingMm));
  holesPanel.redraw(); resPanel.redraw();
}));


// ── Step 1: calibration canvas ───────────────────────────────────────────
function drawCalCanvas(ctx, k, panel) {
  drawImage(ctx);

  const dot = (x, y, color, label) => {
    const r = 8 * k;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    if (label) {
      ctx.font = `bold ${14 * k}px sans-serif`;
      ctx.lineWidth = 3 * k; ctx.strokeStyle = 'rgba(0,0,0,0.75)';
      ctx.strokeText(label, x + r + 3 * k, y + 5 * k);
      ctx.fillStyle = '#fff'; ctx.fillText(label, x + r + 3 * k, y + 5 * k);
    }
  };
  const transLine = (l, color, label) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2 * k;
    line(ctx, l.a, l.b);
    dot(l.a.x, l.a.y, color, ''); dot(l.b.x, l.b.y, color, '');
    dot((l.a.x + l.b.x) / 2, (l.a.y + l.b.y) / 2, color, label);
  };

  if (_tapePoints.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(_tapePoints[0].x, _tapePoints[0].y);
    for (let i = 1; i < _tapePoints.length; i++) ctx.lineTo(_tapePoints[i].x, _tapePoints[i].y);
    ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2 * k;
    ctx.setLineDash([8 * k, 4 * k]); ctx.stroke(); ctx.setLineDash([]);
  }
  _tapePoints.forEach((pt, i) => dot(pt.x, pt.y, '#00e5ff', String(i + 1)));

  if (_mpPoint) drawCross(ctx, _mpPoint.x, _mpPoint.y, 16 * k, 3 * k, '#ffd700', 'rgba(0,0,0,0.6)');

  if (_mpPoint && (_tapePoints.length >= 2 || (_transL1 && _transL2))) {
    const { axisDx: adx, axisDy: ady } = computeAxisAndMount();
    const len = 100 * k, hw = 12 * k;
    const ox = _mpPoint.x, oy = _mpPoint.y, ex = ox + adx * len, ey = oy + ady * len;
    ctx.strokeStyle = ctx.fillStyle = '#ff6600';
    ctx.lineWidth = 3 * k; ctx.lineCap = 'round';
    line(ctx, { x: ox, y: oy }, { x: ex, y: ey });
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - adx * hw * 1.5 - ady * hw, ey - ady * hw * 1.5 + adx * hw);
    ctx.lineTo(ex - adx * hw * 1.5 + ady * hw, ey - ady * hw * 1.5 - adx * hw);
    ctx.closePath(); ctx.fill();
    ctx.font = `bold ${12 * k}px sans-serif`;
    ctx.fillText('→TIP', ex + adx * (hw + 2 * k), ey + ady * (hw + 2 * k));
  }

  if (_transL1)      transLine(_transL1, '#ff55ff', 'L1');
  if (_transL1Start) dot(_transL1Start.x, _transL1Start.y, '#ff55ff', 'L1…');
  if (_transL2)      transLine(_transL2, '#cc44cc', 'L2');
  if (_transL2Start) dot(_transL2Start.x, _transL2Start.y, '#cc44cc', 'L2…');

  if (_transL1 && _transL2) {
    const midA = { x: (_transL1.a.x + _transL1.b.x) / 2, y: (_transL1.a.y + _transL1.b.y) / 2 };
    const midB = { x: (_transL2.a.x + _transL2.b.x) / 2, y: (_transL2.a.y + _transL2.b.y) / 2 };
    const dx = midB.x - midA.x, dy = midB.y - midA.y, len = Math.hypot(dx, dy) || 1;
    const ext = Math.max(_imgW, _imgH);
    ctx.strokeStyle = 'rgba(255,100,255,0.65)'; ctx.lineWidth = 2 * k;
    ctx.setLineDash([8 * k, 4 * k]);
    line(ctx, { x: midA.x - dx / len * ext, y: midA.y - dy / len * ext }, { x: midA.x + dx / len * ext, y: midA.y + dy / len * ext });
    ctx.setLineDash([]);
  }

  // Preview of the mark being placed (touch drag)
  const pv = panel.preview;
  if (pv && _calMode) {
    ctx.save();
    ctx.globalAlpha = 0.75;
    if (_calMode === 'tape') {
      dot(pv.x, pv.y, '#00e5ff', String(_tapePoints.length + 1));
    } else if (_calMode === 'mp') {
      drawCross(ctx, pv.x, pv.y, 16 * k, 3 * k, '#ffd700', 'rgba(0,0,0,0.6)');
    } else {
      const col = _calMode === 'trans1' ? '#ff55ff' : '#cc44cc';
      const start = _calMode === 'trans1' ? _transL1Start : _transL2Start;
      if (start) { ctx.strokeStyle = col; ctx.lineWidth = 2 * k; line(ctx, start, pv); }
      dot(pv.x, pv.y, col, '');
    }
    ctx.restore();
  }

  updateCalStatus();
}

function updateCalStatus() {
  const el = $('cal-status');
  if (_tapePoints.length < 2) { el.textContent = ''; return; }
  const d0 = Math.hypot(_tapePoints[1].x - _tapePoints[0].x, _tapePoints[1].y - _tapePoints[0].y);
  const refMm = parseFloat($('ref-mm').value) || 100;
  el.textContent = `${_tapePoints.length} mark(s) · first interval ${d0.toFixed(0)} px = ${refMm} mm → ~${(refMm / d0).toFixed(4)} mm/px` +
    (_tapePoints.length >= 3 ? ' · parallax correction enabled' : ' · add a 3rd mark for parallax correction');
}

function onCalTap(pos, panel) {
  const { x, y } = pos;
  const thr = 20 / panel.scale;
  const near = p => Math.hypot(p.x - x, p.y - y) < thr;
  const lbl = $('cal-mode-label');

  if (_calMode === 'tape') {
    const idx = _tapePoints.findIndex(near);
    if (idx >= 0) {
      _tapePoints.splice(idx, 1);
      lbl.textContent = `Mark removed — ${_tapePoints.length} tape mark(s) set.`;
    } else {
      _tapePoints.push({ x, y });
      const n = _tapePoints.length;
      if (n >= 4) {
        _calMode = null; updateCalButtons();
        lbl.textContent = '✓ 4 tape marks set. Now click "② Mount point".';
      } else {
        lbl.textContent = `Mark ${n} set — tap next${n >= 2 ? ' or switch tool' : ''} (${4 - n} more for best correction).`;
      }
    }
  } else if (_calMode === 'mp') {
    if (_mpPoint && near(_mpPoint)) {
      _mpPoint = null;
      lbl.textContent = 'Mount point removed — tap to set it again.';
    } else {
      _mpPoint = { x, y };
      _calMode = null; updateCalButtons();
      lbl.textContent = '✓ Mounting point set. Click "Calibrate →", or set axis lines ③④.';
    }
  } else if (_calMode === 'trans1') {
    if (!_transL1Start) {
      if (_transL1 && (near(_transL1.a) || near(_transL1.b))) {
        _transL1 = null; lbl.textContent = 'L1 removed — tap one ski edge to start again.';
      } else {
        _transL1Start = { x, y }; lbl.textContent = 'L1: now tap the OPPOSITE ski edge.';
      }
    } else {
      _transL1 = { a: _transL1Start, b: { x, y } };
      _transL1Start = null; _calMode = null; updateCalButtons();
      lbl.textContent = '✓ L1 set. Now click "④ Axis L2" (tip side).';
    }
  } else if (_calMode === 'trans2') {
    if (!_transL2Start) {
      if (_transL2 && (near(_transL2.a) || near(_transL2.b))) {
        _transL2 = null; lbl.textContent = 'L2 removed — tap one ski edge to start again.';
      } else {
        _transL2Start = { x, y }; lbl.textContent = 'L2: now tap the OPPOSITE ski edge.';
      }
    } else {
      _transL2 = { a: _transL2Start, b: { x, y } };
      _transL2Start = null; _calMode = null; updateCalButtons();
      lbl.textContent = '✓ Centreline from L1 & L2. Click "Calibrate →".';
    }
  }
  updateCalibrateButton();
}

function startCalMode(mode) {
  if (!_img) { showError('Choose a photo first.'); return; }
  _calMode = mode;
  updateCalButtons();
  const lbl = $('cal-mode-label');
  if (mode === 'tape') {
    _tapePoints = []; _axisFlipped = false;
    $('btn-flip').classList.remove('active');
    lbl.textContent = 'Click the FIRST tape mark (heel side). Then click 1–3 more marks at equal spacing toward the tip.';
  } else if (mode === 'mp') {
    lbl.textContent = 'Click the ski CENTRE MARK (half-sole / mounting reference line on ski).';
  } else if (mode === 'trans1') {
    _transL1 = null; _transL1Start = null;
    lbl.textContent = 'L1 (heel side): click one ski EDGE, then click the OPPOSITE edge.';
  } else if (mode === 'trans2') {
    _transL2 = null; _transL2Start = null;
    lbl.textContent = 'L2 (tip side): click one ski EDGE, then click the OPPOSITE edge.';
  }
  updateCalibrateButton();
  // On a phone the tool buttons sit below the photo: bring the photo back into view.
  $('cal-panel').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  calPanel.redraw();
}

document.querySelectorAll('[data-cal-mode]').forEach(btn => btn.addEventListener('click', () => startCalMode(btn.dataset.calMode)));
$('btn-flip').addEventListener('click', () => {
  _axisFlipped = !_axisFlipped;
  $('btn-flip').classList.toggle('active', _axisFlipped);
  calPanel.redraw();
});
$('ref-mm').addEventListener('input', () => calPanel.redraw());

function updateCalButtons() {
  for (const [id, mode] of [['btn-tape', 'tape'], ['btn-mp', 'mp'], ['btn-trans1', 'trans1'], ['btn-trans2', 'trans2']]) {
    $(id).classList.toggle('active', _calMode === mode);
  }
}

function updateCalibrateButton() {
  $('calibrate-btn').disabled = !(_tapePoints.length >= 2 && _mpPoint);
}

/** Ski axis direction and mount point (projected on the centreline when L1/L2 are set). */
function computeAxisAndMount() {
  let axisDx, axisDy;
  let mpX = _mpPoint ? _mpPoint.x : 0, mpY = _mpPoint ? _mpPoint.y : 0;

  if (_transL1 && _transL2) {
    const midA = { x: (_transL1.a.x + _transL1.b.x) / 2, y: (_transL1.a.y + _transL1.b.y) / 2 };
    const midB = { x: (_transL2.a.x + _transL2.b.x) / 2, y: (_transL2.a.y + _transL2.b.y) / 2 };
    const dx = midB.x - midA.x, dy = midB.y - midA.y, len = Math.hypot(dx, dy) || 1;
    axisDx = dx / len; axisDy = dy / len;
    if (_mpPoint) {
      const t = (_mpPoint.x - midA.x) * axisDx + (_mpPoint.y - midA.y) * axisDy;
      mpX = midA.x + t * axisDx; mpY = midA.y + t * axisDy;
    }
  } else if (_tapePoints.length >= 2) {
    const first = _tapePoints[0], last = _tapePoints[_tapePoints.length - 1];
    const dx = last.x - first.x, dy = last.y - first.y, len = Math.hypot(dx, dy) || 1;
    axisDx = dx / len; axisDy = dy / len;
  } else {
    axisDx = -1.0; axisDy = 0.0;
  }
  if (_axisFlipped) { axisDx = -axisDx; axisDy = -axisDy; }
  return { axisDx, axisDy, mpX, mpY };
}

$('calibrate-btn').addEventListener('click', () => {
  if (!_img)                  { showError('No image loaded.'); return; }
  if (_tapePoints.length < 2) { showError('Click at least 2 tape marks first.'); return; }
  if (!_mpPoint)              { showError('Click the mounting point on the ski.'); return; }
  const refMm = parseFloat($('ref-mm').value);
  if (!(refMm > 0))           { showError('Enter a valid mark spacing.'); return; }

  const { axisDx, axisDy, mpX, mpY } = computeAxisAndMount();
  try {
    _cal = Calib.build({
      tape: _tapePoints, spacingMm: refMm,
      mount: { x: mpX, y: mpY }, axis: { dx: axisDx, dy: axisDy },
      imageWidth: _imgW, imageHeight: _imgH,
    });
  } catch (err) {
    showError(err.message);
    return;
  }
  clearError();
  _calMode = null; updateCalButtons();
  renderCalSummary();
  showStep(2);
});

function renderCalSummary() {
  const c = _cal;
  $('cal-summary').textContent = `Scale: ${c.mmPerPx.toFixed(4)} mm/px · ` +
    (c.corrected ? `parallax corrected (${c.nMarks} marks)` : 'no parallax correction');
  const warn = [];
  if (!c.corrected) {
    warn.push('Only 2 tape marks: perspective is not corrected. A camera tilted 10° along the ski then puts the binding holes off by several millimetres — add a third mark.');
  }
  if (c.scaleSpread > 0.12) {
    warn.push(`Strong perspective: the scale changes by ${(c.scaleSpread * 100).toFixed(0)} % along the tape. Lengthwise this is corrected, sideways only approximately — a photo taken more squarely above the ski will be more accurate.`);
  }
  if (c.residualMm > 0.7) {
    warn.push(`The tape marks do not line up with any one perspective (off by ${c.residualMm.toFixed(1)} mm) — one of them is probably misplaced.`);
  }
  const el = $('cal-warning');
  el.style.display = warn.length ? '' : 'none';
  el.innerHTML = warn.map(w => `<div>⚠ ${esc(w)}</div>`).join('');
}


// ── Step 2: mark holes ───────────────────────────────────────────────────
function drawHolesCanvas(ctx, k, panel) {
  drawImage(ctx);
  if (!_cal) return;
  if (_gridEnabled) drawGrid(ctx, k);
  drawCross(ctx, _cal.mount.x, _cal.mount.y, 16 * k, 3 * k, '#ffd700', 'rgba(0,0,0,0.6)');

  const ring = (p, dashed) => {
    const r = Math.max(holeRadiusPx(Calib.toMm(_cal, p).x, k, 0) * 1.7, 11 * k);
    ctx.lineWidth = 2.5 * k;
    ctx.setLineDash(dashed ? [4 * k, 4 * k] : []);
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.beginPath(); ctx.arc(p.x, p.y, r + 1.5 * k, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = 'rgba(50,255,100,0.95)';
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    // the exact marked point
    ctx.fillStyle = 'rgba(50,255,100,0.95)';
    ctx.beginPath(); ctx.arc(p.x, p.y, 1.6 * k, 0, Math.PI * 2); ctx.fill();
  };
  _holes.forEach(h => ring(h, false));
  if (panel.preview) { ctx.save(); ctx.globalAlpha = 0.7; ring(panel.preview, true); ctx.restore(); }

  const n = _holes.length;
  $('holes-count-label').textContent = n === 0 ? 'No holes marked' : `${n} hole(s)`;
}

function onHoleTap(pos, panel) {
  // Tapping on a mark removes it. "On" is measured on the ski, not on the screen: at most
  // 4 mm, i.e. the hole itself. Holes from two old mounts can sit 5 mm apart, and a wider
  // reach would silently delete one while marking the other — a missing hole is the one
  // error that turns a real conflict into "OK". Zoom in, or use Undo, to remove a mark.
  const pxPerMm = Calib.pxPerMm(_cal, Calib.toMm(_cal, pos).x);
  const reachMm = clamp(10 / panel.scale / pxPerMm, 2.5, 4);
  const reachPx = reachMm * pxPerMm;
  let best = -1, bestD = reachPx;
  _holes.forEach((h, i) => {
    const d = Math.hypot(h.x - pos.x, h.y - pos.y);
    if (d < bestD) { bestD = d; best = i; }
  });
  if (best >= 0) _holes.splice(best, 1);
  else _holes.push({ x: pos.x, y: pos.y });
}

$('btn-clear-holes').addEventListener('click', () => {
  if (_holes.length > 3 && !confirm(`Remove all ${_holes.length} hole marks?`)) return;
  _holes = []; holesPanel.redraw();
});
$('btn-undo-hole').addEventListener('click', () => { _holes.pop(); holesPanel.redraw(); });
$('btn-back-cal').addEventListener('click', () => showStep(1));


// ── Step 2 → 3: analysis ─────────────────────────────────────────────────
function readParams() {
  const bsl = parseFloat($('bsl-input').value);
  const minSep = parseFloat($('sep-input').value);
  if (!(bsl >= 200 && bsl <= 420)) throw new Error('Enter a boot sole length between 200 and 420 mm.');
  if (!(minSep >= 0 && minSep <= 50)) throw new Error('Enter a minimum hole separation between 0 and 50 mm.');
  return { bsl, minSep, category: $('cat-select').value };
}

$('analyze-btn').addEventListener('click', () => {
  if (!_cal) { showError('Complete calibration first.'); return; }
  if (!_db)  { showError('Binding templates are not loaded.'); return; }
  let p;
  try { p = readParams(); } catch (err) { showError(err.message); return; }
  clearError();

  _existing = _holes.map(h => Calib.toMm(_cal, h));
  _results = Matcher.rankBindings(_db, p.bsl, _existing, { minSep: p.minSep, category: p.category });
  _identified = null; _identShown = null; _clearCache = null;
  renderIdentify();
  showStep(3);
  renderResults(p);
});

function statusBadge(r) {
  if (!r.mountable) return '<span class="badge badge-err">CONFLICT</span>';
  if (r.tight) return '<span class="badge badge-warn" title="Mountable, but within 2 mm of your limit">TIGHT</span>';
  return '<span class="badge badge-ok">OK</span>';
}

function renderResults(p) {
  $('hole-count').textContent = `${_existing.length} existing hole(s) · scale ${_cal.mmPerPx.toFixed(4)} mm/px`;
  const tbody = $('results-body');
  tbody.innerHTML = '';
  $('results-table-card').style.display = 'block';

  const skipped = _db.bindings.filter(b =>
    (p.category === 'all' || b.category === p.category) && !Matcher.bslInRange(b, p.bsl));

  if (!_results.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="table-note">No binding in the database supports this boot sole length.</td></tr>';
    _sel = null; _view = null;
    $('overlay-controls-card').style.display = 'none';
    renderSummary();
    resPanel.redraw();
    return;
  }

  for (const r of _results) {
    const tr = document.createElement('tr');
    tr.dataset.id = r.bindingId;
    const heel = r.heelOffset !== 0 ? `${signed(r.heelOffset)} mm` : '—';
    const clearance = r.minClearance == null ? '—' : `${r.minClearance.toFixed(1)} mm`;
    const unverified = r.verified ? '' : ' <span class="badge badge-warn" title="Unverified template">⚠</span>';
    tr.innerHTML = `
      <td>${esc(r.bindingName)}${unverified}</td>
      <td>${r.nNew}</td>
      <td>${r.nReusable}</td>
      <td>${r.nConflicts}</td>
      <td class="cell-dim">${heel}</td>
      <td class="cell-dim">${clearance}</td>
      <td>${statusBadge(r)}</td>`;
    tr.addEventListener('click', () => selectBinding(r.bindingId));
    tbody.appendChild(tr);
  }
  if (skipped.length) {
    const tr = document.createElement('tr');
    tr.className = 'no-select';
    tr.innerHTML = `<td colspan="7" class="table-note">Not shown — boot sole length out of range: ${esc(skipped.map(b => b.name).join(', '))}</td>`;
    tbody.appendChild(tr);
  }
  selectBinding(_results[0].bindingId);
}


// ── Step 3: selected binding ─────────────────────────────────────────────
function setSlider(id, value, min, max, step) {
  for (const el of [$(id), $(id + '-num')]) { el.min = min; el.max = max; el.step = step; el.value = value; }
}

function selectBinding(id) {
  const binding = _db.bindings.find(b => b.id === id);
  if (!binding) return;
  document.querySelectorAll('#results-body tr').forEach(tr => tr.classList.toggle('selected', tr.dataset.id === id));
  _sel = { binding, heelAuto: true };

  const p = readParams();
  const range = binding.bsl_range_mm;
  setSlider('ctrl-bsl', p.bsl, range ? range[0] : 240, range ? range[1] : 400, 1);
  const heelRange = (binding.heel_unit && binding.heel_unit.adjustment_range_mm) || 0;
  $('ctrl-heel-group').style.display = heelRange > 0 ? '' : 'none';
  setSlider('ctrl-heel', 0, -heelRange, heelRange, 0.5);
  setSlider('ctrl-mount', 0, -30, 30, 0.5);
  $('overlay-controls-card').style.display = '';
  updateView();
}

for (const id of ['ctrl-bsl', 'ctrl-heel', 'ctrl-mount']) {
  const rng = $(id), num = $(id + '-num');
  const changed = from => {
    (from === rng ? num : rng).value = from.value;
    if (id === 'ctrl-heel' && _sel) _sel.heelAuto = false;
    updateView();
  };
  rng.addEventListener('input', () => changed(rng));
  num.addEventListener('input', () => changed(num));
}
$('btn-heel-auto').addEventListener('click', () => { if (_sel) { _sel.heelAuto = true; updateView(); } });

function updateView() {
  if (!_sel) return;
  const p = readParams();
  const num = (id, fallback) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : fallback; };
  const bsl = num('ctrl-bsl', p.bsl);
  const mountX = clamp(num('ctrl-mount', 0), -30, 30);
  try {
    _view = Matcher.checkBinding(_sel.binding, bsl, _existing, {
      minSep: p.minSep, mountX,
      heelOffset: _sel.heelAuto ? null : num('ctrl-heel', 0),
    });
  } catch (err) {
    _view = null;
    $('sum-name').textContent = _sel.binding.name;
    $('sum-status').innerHTML = `<span class="badge badge-warn">${esc(err.message)}</span>`;
    $('sum-legend').innerHTML = $('sum-detail').innerHTML = '';
    $('sum-hint').style.display = 'none';
    resPanel.redraw();
    return;
  }
  if (_sel.heelAuto) { $('ctrl-heel').value = $('ctrl-heel-num').value = _view.heelOffset; }
  $('btn-heel-auto').classList.toggle('active', _sel.heelAuto);
  renderSummary();
  resPanel.redraw();
}

const holeName = h => (h.label || '').replace(/^(front|heel)_/, '').replace(/_/g, ' ');
const unitName = h => (h.unit === 'front' ? 'toe' : 'heel');

function renderSummary() {
  const v = _view;
  if (!v) {
    $('sum-name').textContent = 'No binding selected';
    $('sum-status').innerHTML = $('sum-legend').innerHTML = $('sum-detail').innerHTML = '';
    $('sum-hint').style.display = 'none';
    return;
  }
  const p = readParams();
  $('sum-name').textContent = v.bindingName;
  $('sum-status').innerHTML = statusBadge(v) + ` <span class="sum-sub">boot sole ${v.bslMm} mm${v.mountX ? ` · mounted ${signed(v.mountX)} mm from the mark` : ''}</span>`;

  const sw = (cls, text) => `<span class="legend-item"><span class="swatch ${cls}"></span>${text}</span>`;
  $('sum-legend').innerHTML =
    sw('sw-existing', `${_existing.length} existing`) +
    sw('sw-reusable', `${v.nReusable} reused`) +
    sw('sw-new', `${v.nNew} to drill`) +
    sw('sw-conflict', `${v.nConflicts} conflict${v.nConflicts === 1 ? '' : 's'}`);

  const lines = [];
  for (const h of v.holes.filter(h => h.status === 'conflict')) {
    lines.push(`<div class="d-err">✕ ${unitName(h)}, ${esc(holeName(h))}: ${h.distance.toFixed(1)} mm from an existing hole (limit ${p.minSep} mm)</div>`);
  }
  if (v.mountable && v.minClearance != null) {
    lines.push(`<div>Closest hole to drill is ${v.minClearance.toFixed(1)} mm from an existing hole (limit ${p.minSep} mm).</div>`);
    if (v.tight) lines.push('<div class="d-warn">⚠ That is within 2 mm of your limit — closer than a photo can measure. Check with the printed 1:1 sheet.</div>');
  }
  if (!_existing.length) lines.push('<div>No existing holes marked — every hole is new.</div>');
  const loose = v.holes.filter(h => h.status === 'reusable' && h.distance > Matcher.LOOSE_REUSE_MM);
  if (loose.length) {
    lines.push(`<div class="d-warn">⚠ ${loose.length} reused hole(s) are 1–2 mm off the template. If that offset is real rather than marking error, the hole cannot be reused — check on the ski.</div>`);
  }
  if (v.heelOffset) {
    lines.push(`<div>Heel piece ${Math.abs(v.heelOffset).toFixed(1)} mm ${v.heelOffset > 0 ? 'forward' : 'back'} on its track (range ±${v.heelRange} mm)${_sel.heelAuto ? ', chosen automatically' : ''}.</div>`);
  }
  $('sum-detail').innerHTML = lines.join('');

  // If it conflicts here: how far would the whole binding have to move?
  const hint = $('sum-hint');
  if (v.nConflicts > 0) {
    const key = [v.bindingId, v.bslMm, p.minSep].join('|');
    if (!_clearCache || _clearCache.key !== key) {
      _clearCache = { key, result: Matcher.findClearMountOffset(_sel.binding, v.bslMm, _existing, { minSep: p.minSep }) };
    }
    const c = _clearCache.result;
    hint.style.display = '';
    if (c && Math.abs(c.mountX - v.mountX) > 1e-9) {
      const where = c.mountX === 0 ? 'on the mount mark' : `${Math.abs(c.mountX).toFixed(1)} mm ${c.mountX > 0 ? 'forward of' : 'behind'} the mount mark`;
      hint.innerHTML = `Nearest conflict-free position: ${where}` +
        (c.heelOffset ? `, heel ${signed(c.heelOffset)} mm` : '') +
        ` <button class="btn-sm" id="btn-apply-clear">Show</button>`;
      $('btn-apply-clear').addEventListener('click', () => {
        _sel.heelAuto = true;
        $('ctrl-mount').value = $('ctrl-mount-num').value = c.mountX;
        updateView();
      });
    } else if (c) {
      hint.textContent = 'The heel track alone can clear this — switch the heel offset back to auto.';
    } else {
      hint.textContent = 'No conflict-free position within ±30 mm of the mount mark, even using the heel track.';
    }
  } else {
    hint.style.display = 'none';
  }
}

function drawResCanvas(ctx, k) {
  drawImage(ctx);
  if (!_cal) return;
  if (_gridEnabled) drawGrid(ctx, k);

  // Centreline
  const ext = Math.hypot(_imgW, _imgH);
  ctx.strokeStyle = 'rgba(0,220,220,0.65)'; ctx.lineWidth = 2 * k;
  line(ctx,
    { x: _cal.mount.x - _cal.axis.dx * ext, y: _cal.mount.y - _cal.axis.dy * ext },
    { x: _cal.mount.x + _cal.axis.dx * ext, y: _cal.mount.y + _cal.axis.dy * ext });

  const disc = (mm, fill, minPx) => {
    const p = Calib.toPx(_cal, mm), r = holeRadiusPx(mm.x, k, minPx);
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5 * k; ctx.stroke();
    return { p, r };
  };

  _existing.forEach(e => disc(e, '#ffffff', 4.5));

  if (_identShown) {
    ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2.5 * k;
    ctx.setLineDash([5 * k, 4 * k]);
    for (const h of _identShown.holes) {
      const p = Calib.toPx(_cal, h), r = holeRadiusPx(h.x, k, 4.5) * 2.2;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  if (_view) {
    const colour = { reusable: '#32cd32', new: '#ffa500', conflict: '#dc143c' };
    for (const h of _view.holes) {
      const { p, r } = disc(h, colour[h.status], 5.5);
      if (h.status === 'conflict') {
        ctx.strokeStyle = '#000'; ctx.lineWidth = 2 * k;
        line(ctx, { x: p.x - r, y: p.y - r }, { x: p.x + r, y: p.y + r });
        line(ctx, { x: p.x + r, y: p.y - r }, { x: p.x - r, y: p.y + r });
      }
    }
  }

  // Yellow cross: the ski's mount mark. White cross: where the boot centre sits when offset.
  drawCross(ctx, _cal.mount.x, _cal.mount.y, 16 * k, 3 * k, '#ffd700', 'rgba(0,0,0,0.6)');
  if (_view && Math.abs(_view.mountX) >= 0.1) {
    const p = Calib.toPx(_cal, { x: _view.mountX, y: 0 });
    drawCross(ctx, p.x, p.y, 13 * k, 3 * k, '#ffffff', '#000000');
  }
}

$('btn-edit-holes').addEventListener('click', () => showStep(2));


// ── Step 3: previously mounted binding ───────────────────────────────────
$('btn-identify').addEventListener('click', () => {
  if (!_db || !_cal) return;
  _identified = Matcher.identify(_db, _existing);
  _identShown = _identified.full[0] || null;
  renderIdentify();
  resPanel.redraw();
});

function renderIdentify() {
  const body = $('identify-body');
  if (!_identified) {
    body.innerHTML = '<p class="ident-note">Compares the marked holes with every template to find the binding that made them, the boot sole length it was set for, and where it sat.</p>';
    return;
  }
  const { full, partial } = _identified;
  if (_existing.length < 3) {
    body.innerHTML = '<p class="ident-note">Mark the existing holes first — a pattern needs at least three.</p>';
    return;
  }
  if (!full.length && !partial.length) {
    body.innerHTML = `<p class="ident-note">No known pattern in these holes. The old binding may not be among the ${_db.bindings.length} templates, or a hole is missing or misplaced.</p>`;
    return;
  }

  const fit = rms => `fit ±${rms.toFixed(1)} mm`;
  const where = x => Math.abs(x) < 1 ? 'on your mount mark' : `${Math.abs(x).toFixed(1)} mm ${x > 0 ? 'forward of' : 'behind'} your mount mark`;
  const rows = [];

  if (full.length) {
    const best = full[0];
    rows.push(best.rms <= 1
      ? `<p class="ident-verdict ok">✓ The holes match a known pattern to within ${Math.max(best.rms, 0.1).toFixed(1)} mm, so your scale and axis look right.</p>`
      : `<p class="ident-verdict warn">⚠ A pattern is recognisable but only loosely (±${best.rms.toFixed(1)} mm): the calibration or some hole marks are a little off.</p>`);
  }
  full.forEach((m, i) => {
    const lines = [`all ${m.matched} holes · ${fit(m.rms)}`];
    if (m.bslEst != null) {
      lines.push(`boot sole ≈ ${m.bslEst.toFixed(0)} mm${m.heelRange ? ` (±${m.heelRange} mm, depending on where the heel sat on its track)` : ''}`);
    }
    lines.push(`centred ${where(m.mountOffset)}`);
    if (Math.abs(m.lateralOffset) > 1.5) lines.push(`${Math.abs(m.lateralOffset).toFixed(1)} mm off your centreline`);
    rows.push(`<div class="ident-row${_identShown === m ? ' shown' : ''}" data-kind="full" data-i="${i}">
      <strong>${esc(m.bindingName)}</strong><span>${lines.map(esc).join(' · ')}</span></div>`);
  });
  if (partial.length) {
    rows.push(`<p class="ident-note">${full.length ? 'Other partial matches' : 'Only one unit of a known pattern was found'}:</p>`);
    partial.forEach((m, i) => {
      const names = [m.bindingName, ...m.alsoFits].map(esc).join(' / ');
      rows.push(`<div class="ident-row${_identShown === m ? ' shown' : ''}" data-kind="partial" data-i="${i}">
        <strong>${names}</strong><span>${m.kind} piece only (${m.matched} holes) · ${fit(m.rms)}</span></div>`);
    });
  }
  rows.push('<p class="ident-note">Tap a match to outline it on the photo (dashed cyan).</p>');
  body.innerHTML = rows.join('');
  body.querySelectorAll('.ident-row').forEach(el => el.addEventListener('click', () => {
    const m = _identified[el.dataset.kind][Number(el.dataset.i)];
    _identShown = _identShown === m ? null : m;
    renderIdentify();
    resPanel.redraw();
  }));
}


// ── Step 3: printable 1:1 check sheet ────────────────────────────────────
$('btn-print').addEventListener('click', () => {
  if (!_view) { showError('Select a binding first.'); return; }
  const p = readParams();
  const sheet = PrintSheet.build({
    bindingName: _view.bindingName, bslMm: _view.bslMm, heelOffset: _view.heelOffset, mountX: _view.mountX,
    minSep: p.minSep, date: new Date().toISOString().slice(0, 10),
    holes: _view.holes, existing: _existing, paper: $('paper-select').value,
  });
  let style = $('print-page-style');
  if (!style) { style = document.createElement('style'); style.id = 'print-page-style'; document.head.appendChild(style); }
  style.textContent = `@page { size: ${sheet.paper.w}mm ${sheet.paper.h}mm; margin: 0; }`;
  $('print-root').innerHTML = sheet.pages
    .map(svg => `<div class="print-page" style="width:${sheet.paper.w}mm;height:${sheet.paper.h}mm">${svg}</div>`).join('');
  window.print();
});


// ── Step navigation ──────────────────────────────────────────────────────
function showStep(n) {
  for (let i = 1; i <= 3; i++) {
    $('step' + i).style.display = i === n ? '' : 'none';
    const nav = $('snav-' + i);
    nav.classList.toggle('done', i < n);
    nav.classList.toggle('active', i === n);
  }
  PANELS[['cal', 'holes', 'res'][n - 1]].setZoom(1);
}

[1, 2, 3].forEach(n => $('snav-' + n).addEventListener('click', () => {
  if ($('snav-' + n).classList.contains('done')) showStep(n);
}));


// ── Messages ─────────────────────────────────────────────────────────────
function showError(msg) {
  const el = $('error-notice');
  el.textContent = msg; el.style.display = 'block';
}
function clearError() { $('error-notice').style.display = 'none'; }
