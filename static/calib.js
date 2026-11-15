/* calib.js — photo ↔ ski coordinate conversion.
 *
 * Pure geometry, no DOM: runs in the browser (window.Calib) and under Node for tests.
 *
 * Ski frame (mm): origin at the mount mark, x toward the tip, y toward the skier's
 * right as seen from above. Image frame (px): x right, y down.
 *
 * The photo is never resampled. Perspective along the ski is handled by a 1-D
 * projective map fitted to the tape marks, applied to coordinates instead of pixels.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Calib = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /** Solve a 3×3 linear system by Gaussian elimination with partial pivoting. */
  function solve3(A, b) {
    const m = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < 3; col++) {
      let piv = col;
      for (let r = col + 1; r < 3; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
      if (Math.abs(m[piv][col]) < 1e-12) return null;
      [m[col], m[piv]] = [m[piv], m[col]];
      for (let r = 0; r < 3; r++) {
        if (r === col) continue;
        const f = m[r][col] / m[col][col];
        for (let c = col; c < 4; c++) m[r][c] -= f * m[col][c];
      }
    }
    return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
  }

  /**
   * Fit  target = (p·u + q) / (r·u + 1)  through (uObs[i], uTgt[i]) by linear least squares.
   * Coordinates are normalised first so the normal equations stay well conditioned.
   */
  function fitProjective(uObs, uTgt) {
    const L = Math.max(...uObs.map(Math.abs), ...uTgt.map(Math.abs), 1);
    const AtA = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const Atb = [0, 0, 0];
    for (let i = 0; i < uObs.length; i++) {
      const u = uObs[i] / L, t = uTgt[i] / L;
      const row = [u, -t * u, 1];
      for (let a = 0; a < 3; a++) {
        Atb[a] += row[a] * t;
        for (let c = 0; c < 3; c++) AtA[a][c] += row[a] * row[c];
      }
    }
    const sol = solve3(AtA, Atb);
    if (!sol) return null;
    return { p: sol[0], q: sol[2] * L, r: sol[1] / L };
  }

  /**
   * Build a calibration.
   *
   * @param {object} o
   * @param {{x:number,y:number}[]} o.tape      tape marks (image px), equal real-world spacing
   * @param {number} o.spacingMm                real distance between consecutive marks
   * @param {{x:number,y:number}} o.mount       mount mark (image px)
   * @param {{dx:number,dy:number}} o.axis      heel→tip direction in the image
   * @param {number} [o.imageWidth]             used to locate the optical centre
   * @param {number} [o.imageHeight]
   * @returns calibration object; throws Error with a user-readable message on bad input
   */
  function build(o) {
    const n = o.tape.length;
    if (n < 2) throw new Error('Need at least 2 tape marks.');
    if (!(o.spacingMm > 0)) throw new Error('Enter a valid mark spacing.');
    const len = Math.hypot(o.axis.dx, o.axis.dy);
    if (!(len > 1e-6)) throw new Error('Ski axis direction is undefined.');
    const axis = { dx: o.axis.dx / len, dy: o.axis.dy / len };
    const mount = { x: o.mount.x, y: o.mount.y };

    const uObs = o.tape
      .map(t => (t.x - mount.x) * axis.dx + (t.y - mount.y) * axis.dy)
      .sort((a, b) => a - b);
    for (let i = 1; i < n; i++) {
      if (uObs[i] - uObs[i - 1] < 1) throw new Error('Tape marks coincide along the ski axis — place them further apart.');
    }
    const meanSpan = (uObs[n - 1] - uObs[0]) / (n - 1);
    const mmPerPx = o.spacingMm / meanSpan;

    let proj = { p: 1, q: 0, r: 0 };
    let corrected = false;
    let residualMm = 0;
    if (n >= 3) {
      const uTgt = uObs.map((_, i) => uObs[0] + i * meanSpan);
      const fit = fitProjective(uObs, uTgt);
      if (!fit) throw new Error('Could not fit the tape marks — re-place them.');
      proj = fit;
      corrected = true;
      residualMm = Math.max(...uObs.map((u, i) => Math.abs((fit.p * u + fit.q) / (fit.r * u + 1) - uTgt[i]))) * mmPerPx;
    }

    const cal = {
      mount, axis,
      // skier's right as seen from above (the photo is a top view)
      right: { dx: -axis.dy, dy: axis.dx },
      mmPerPx, proj, corrected, residualMm,
      nMarks: n,
      uCentre: 0,
      uMin: -Infinity, uMax: Infinity,
    };

    // Along-axis coordinate of the image centre: reference point for the lateral scale.
    if (o.imageWidth && o.imageHeight) {
      cal.uCentre = (o.imageWidth / 2 - mount.x) * axis.dx + (o.imageHeight / 2 - mount.y) * axis.dy;
      const us = [[0, 0], [o.imageWidth, 0], [0, o.imageHeight], [o.imageWidth, o.imageHeight]]
        .map(([x, y]) => (x - mount.x) * axis.dx + (y - mount.y) * axis.dy);
      cal.uMin = Math.min(...us);
      cal.uMax = Math.max(...us);
      // The map must stay monotonic over the whole photo, otherwise the marks are inconsistent.
      if (proj.r !== 0) {
        const pole = -1 / proj.r;
        if (pole > cal.uMin - 1 && pole < cal.uMax + 1) {
          throw new Error('Tape marks are inconsistent (spacing changes too abruptly) — re-place them.');
        }
      }
    }
    if (!(proj.p - proj.q * proj.r > 0)) throw new Error('Tape marks are inconsistent — re-place them.');

    // How much the scale changes from one end of the tape to the other (0 = camera square-on).
    cal.scaleSpread = Math.abs(alongScale(cal, uObs[0]) / alongScale(cal, uObs[n - 1]) - 1);
    return cal;
  }

  /** d(target)/d(observed) at along-axis image coordinate u — relative along-axis magnification. */
  function alongScale(cal, u) {
    const { p, q, r } = cal.proj;
    const d = r * u + 1;
    return (p - q * r) / (d * d);
  }

  /**
   * Lateral mm per image px at along-axis coordinate u.
   *
   * For a camera pitched along the ski, along-axis scale varies with depth² and lateral
   * scale with depth, so lateral = sqrt(along(u) · along(optical centre)) up to cos(pitch),
   * which cannot be recovered from collinear marks and is taken as 1.
   */
  function lateralMmPerPx(cal, u) {
    if (!cal.corrected) return cal.mmPerPx;
    return cal.mmPerPx * Math.sqrt(alongScale(cal, u) * alongScale(cal, cal.uCentre));
  }

  /** Image px → ski mm. */
  function toMm(cal, pt) {
    const vx = pt.x - cal.mount.x, vy = pt.y - cal.mount.y;
    const u = vx * cal.axis.dx + vy * cal.axis.dy;
    const v = vx * cal.right.dx + vy * cal.right.dy;
    const { p, q, r } = cal.proj;
    return {
      x: ((p * u + q) / (r * u + 1) - q) * cal.mmPerPx,
      y: v * lateralMmPerPx(cal, u),
    };
  }

  /** Along-axis image coordinate (px from the mount mark) of ski position xMm. */
  function uOf(cal, xMm) {
    const { p, q, r } = cal.proj;
    const t = xMm / cal.mmPerPx + q;
    return (q - t) / (r * t - p);
  }

  /** Ski mm → image px. */
  function toPx(cal, mm) {
    const u = uOf(cal, mm.x);
    const v = mm.y / lateralMmPerPx(cal, u);
    return {
      x: cal.mount.x + u * cal.axis.dx + v * cal.right.dx,
      y: cal.mount.y + u * cal.axis.dy + v * cal.right.dy,
    };
  }

  /** Image px per mm around ski position xMm (geometric mean of both directions) — for drawing sizes. */
  function pxPerMm(cal, xMm) {
    const u = uOf(cal, xMm);
    const along = 1 / (cal.mmPerPx * alongScale(cal, u));
    const lateral = 1 / lateralMmPerPx(cal, u);
    return Math.sqrt(along * lateral);
  }

  return { build, toMm, toPx, uOf, pxPerMm, alongScale, lateralMmPerPx, fitProjective };
});
