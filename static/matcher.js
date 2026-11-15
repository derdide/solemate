/* matcher.js — binding templates vs. existing holes.
 *
 * Pure geometry, no DOM: runs in the browser (window.Matcher) and under Node for tests.
 *
 * All coordinates are ski-frame mm: origin at the mount mark, x toward the tip,
 * y toward the skier's right.
 *
 * binding_db.json convention: hole x values grow toward the TAIL (the drill-template
 * convention). For "half_bsl" bindings they are measured from the boot tip (front unit)
 * and from the boot heel (heel unit); for "fixed" bindings from the mount point.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Matcher = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const DEFAULT_MIN_SEPARATION_MM = 14.0;  // minimum centre-to-centre distance between holes
  const REUSE_TOLERANCE_MM = 2.0;          // centres closer than this are the same hole
  const TIGHT_MARGIN_MM = 2.0;             // "mountable, but within measurement error of a conflict"
  const LOOSE_REUSE_MM = 1.0;              // a "reused" hole further off than this needs a look
  const HEEL_STEP_MM = 0.5;

  const dist = (a, b) => Math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
  /** Largest distance between a reused hole and the existing hole it lands on (µm-rounded), 0 if none. */
  const reuseMisfit = list => {
    let m = 0;
    for (const a of list) if (a.status === 'reusable' && a.distance > m) m = a.distance;
    return Math.floor(m * 1000 + 0.5) / 1000;
  };

  // ── Template → ski coordinates ──────────────────────────────────────────────

  function bslInRange(binding, bslMm) {
    const r = binding.bsl_range_mm;
    return !r || (r[0] <= bslMm && bslMm <= r[1]);
  }

  /**
   * Hole positions of `binding` for boot sole length `bslMm`, with the boot centre
   * at x = mountX. Throws RangeError when the BSL is outside the binding's range.
   */
  function computeHoles(binding, bslMm, mountX = 0) {
    if (!bslInRange(binding, bslMm)) {
      const r = binding.bsl_range_mm;
      throw new RangeError(`BSL ${bslMm} mm is outside ${r[0]}–${r[1]} mm for ${binding.name}`);
    }
    let frontRef, heelRef;
    if (binding.mounting_reference === 'half_bsl') {
      frontRef = mountX + bslMm / 2;   // boot tip
      heelRef = mountX - bslMm / 2;    // boot heel
    } else if (binding.mounting_reference === 'fixed') {
      frontRef = heelRef = mountX;
    } else {
      throw new Error(`Unknown mounting_reference: ${binding.mounting_reference}`);
    }
    const place = (h, unit, ref) => ({ x: ref - h.x, y: h.y, label: h.label, unit });
    return [
      ...binding.front_unit.holes.map(h => place(h, 'front', frontRef)),
      ...binding.heel_unit.holes.map(h => place(h, 'heel', heelRef)),
    ];
  }

  // ── Conflict analysis ───────────────────────────────────────────────────────

  function analyzeHole(hole, existing, minSep, reuseTol) {
    if (!existing.length) return { hole, status: 'new', nearest: null, distance: null };
    let nearest = existing[0], d = dist(hole, nearest);
    for (let i = 1; i < existing.length; i++) {
      const di = dist(hole, existing[i]);
      if (di < d) { d = di; nearest = existing[i]; }
    }
    const status = d <= reuseTol ? 'reusable' : d < minSep ? 'conflict' : 'new';
    return { hole, status, nearest, distance: d };
  }

  const heelRangeOf = binding => (binding.heel_unit && binding.heel_unit.adjustment_range_mm) || 0;

  /**
   * Full analysis of one binding at one BSL.
   *
   * The toe is fixed by the BSL and mount position. The heel unit may sit anywhere on its
   * adjustment track, so unless `heelOffset` is given the track is searched for the position
   * with the fewest conflicts, then the most reused holes, then the best fit on those holes
   * (a hole "reused" 2 mm off-centre is worse than a new one), then the smallest shift.
   *
   * opts: { mountX = 0, minSep = 14, reuseTol = 2, heelOffset = null }
   */
  function checkBinding(binding, bslMm, existing, opts = {}) {
    const mountX = opts.mountX || 0;
    const minSep = opts.minSep ?? DEFAULT_MIN_SEPARATION_MM;
    const reuseTol = opts.reuseTol ?? REUSE_TOLERANCE_MM;
    const holes = computeHoles(binding, bslMm, mountX);
    const front = holes.filter(h => h.unit === 'front');
    const heel = holes.filter(h => h.unit === 'heel');
    const analyse = list => list.map(h => analyzeHole(h, existing, minSep, reuseTol));
    const score = list => [
      list.filter(a => a.status === 'conflict').length,
      list.filter(a => a.status === 'reusable').length,
    ];
    const shifted = off => heel.map(h => ({ ...h, x: h.x + off }));

    const frontAn = analyse(front);
    const range = heelRangeOf(binding);
    let heelOffset = 0;
    let heelAn = analyse(heel);

    if (opts.heelOffset != null) {
      heelOffset = range > 0 ? Math.max(-range, Math.min(range, opts.heelOffset)) : 0;
      heelAn = analyse(shifted(heelOffset));
    } else {
      let [bestC, bestR] = score(heelAn);
      let bestM = reuseMisfit(heelAn);
      if (range > 0 && (bestC > 0 || bestR < heel.length || bestM > 0)) {
        const steps = Math.floor(range / HEEL_STEP_MM);
        for (let s = -steps; s <= steps; s++) {
          const off = s * HEEL_STEP_MM;
          const an = analyse(shifted(off));
          const [c, r] = score(an);
          const m = reuseMisfit(an);
          const better = c !== bestC ? c < bestC
            : r !== bestR ? r > bestR
            : m !== bestM ? m < bestM
            : Math.abs(off) < Math.abs(heelOffset);
          if (better) { bestC = c; bestR = r; bestM = m; heelOffset = off; heelAn = an; }
        }
      }
    }

    const all = [...frontAn, ...heelAn];
    const count = s => all.filter(a => a.status === s).length;
    const nNew = count('new'), nReusable = count('reusable'), nConflicts = count('conflict');
    // Smallest distance from a hole that would be drilled to an existing hole.
    const drilled = all.filter(a => a.status !== 'reusable' && a.distance != null);
    const minClearance = drilled.length ? Math.min(...drilled.map(a => a.distance)) : null;
    const mountable = nConflicts === 0;

    return {
      bindingId: binding.id,
      bindingName: binding.name,
      category: binding.category,
      verified: binding.verified !== false,
      bslMm, mountX, heelOffset, heelRange: range,
      holes: all.map(a => ({ ...a.hole, status: a.status, distance: a.distance, nearest: a.nearest })),
      nNew, nReusable, nConflicts,
      score: nConflicts * 100 + nNew - nReusable * 0.5,
      mountable,
      maxReuseMisfit: reuseMisfit(all),
      minClearance,
      tight: mountable && minClearance != null && minClearance < minSep + TIGHT_MARGIN_MM,
    };
  }

  /** Analyse every binding that supports this BSL; best first. opts: { minSep, category } */
  function rankBindings(db, bslMm, existing, opts = {}) {
    const results = [];
    for (const binding of db.bindings) {
      if (opts.category && opts.category !== 'all' && binding.category !== opts.category) continue;
      if (!bslInRange(binding, bslMm)) continue;
      results.push(checkBinding(binding, bslMm, existing, { minSep: opts.minSep }));
    }
    return results.sort((a, b) => (a.mountable === b.mountable ? a.score - b.score : a.mountable ? -1 : 1));
  }

  /**
   * Smallest mount-point shift (within ±range) for which the binding has no conflict,
   * letting the heel use its track at every position. Since the position is ours to
   * choose, a hole only counts as reused if it lands squarely on the old one. Returns
   * the analysis at that position, or null if there is none.
   * opts: { minSep, range = 30, step = 0.5 }
   */
  function findClearMountOffset(binding, bslMm, existing, opts = {}) {
    const range = opts.range ?? 30, step = opts.step ?? 0.5;
    const n = Math.floor(range / step);
    for (let i = 0; i <= n; i++) {
      for (const sign of i === 0 ? [1] : [1, -1]) {
        const res = checkBinding(binding, bslMm, existing, { minSep: opts.minSep, mountX: sign * i * step });
        if (res.mountable && res.maxReuseMisfit <= LOOSE_REUSE_MM) return res;
      }
    }
    return null;
  }

  // ── Identifying a previously mounted binding ────────────────────────────────

  /**
   * Find every placement of a rigid hole pattern among the existing holes.
   * `pattern` points are relative to the unit's reference (x toward tip). A placement is
   * accepted only if every pattern hole lands on a distinct existing hole.
   * Returns [{ ref, dy, rms, used:[indices into existing] }].
   */
  function fitUnit(pattern, existing, tol, maxLateral) {
    const found = [];
    if (pattern.length < 2) return found;
    const match = (tx, ty, t) => {
      const used = [];
      for (const p of pattern) {
        let best = -1, bestD = t;
        for (let j = 0; j < existing.length; j++) {
          if (used.includes(j)) continue;
          const d = Math.hypot(existing[j].x - (p.x + tx), existing[j].y - (p.y + ty));
          if (d <= bestD) { bestD = d; best = j; }
        }
        if (best < 0) return null;
        used.push(best);
      }
      return used;
    };
    for (let j = 0; j < existing.length; j++) {
      let tx = existing[j].x - pattern[0].x, ty = existing[j].y - pattern[0].y;
      if (Math.abs(ty) > maxLateral + tol) continue;
      // The first guess hangs on a single hole, so it carries that hole's marking error:
      // match loosely, refine the placement on all holes, then apply the real tolerance.
      let used = match(tx, ty, tol * 1.6);
      if (!used) continue;
      // Least-squares translation over the matched pairs, then re-match once.
      const refit = () => {
        tx = used.reduce((s, e, i) => s + existing[e].x - pattern[i].x, 0) / pattern.length;
        ty = used.reduce((s, e, i) => s + existing[e].y - pattern[i].y, 0) / pattern.length;
      };
      refit();
      const again = match(tx, ty, tol);
      if (!again) continue;
      used = again;
      refit();
      if (Math.abs(ty) > maxLateral) continue;
      const rms = Math.sqrt(used.reduce((s, e, i) =>
        s + (existing[e].x - pattern[i].x - tx) ** 2 + (existing[e].y - pattern[i].y - ty) ** 2, 0) / pattern.length);
      const key = [...used].sort((a, b) => a - b).join(',');
      if (!found.some(f => f.key === key)) found.push({ ref: tx, dy: ty, rms, used, key });
    }
    return found;
  }

  /**
   * Match the existing holes against every template to work out which binding was mounted
   * before, at what boot sole length, and where.
   *
   * A match needs every hole of a unit (toe piece or heel piece) to coincide with an
   * existing hole. Toe + heel of the same binding at a plausible distance is a full match;
   * a single unit is reported as partial.
   *
   * opts: { tolerance = 2.5 mm per hole, maxLateral = 6 mm off the centreline, limit = 8 }
   */
  function identify(db, existing, opts = {}) {
    const tol = opts.tolerance ?? 2.5;
    const maxLateral = opts.maxLateral ?? 6;
    const full = [], partial = [];
    const explained = new Set();   // existing holes accounted for by a full match

    for (const binding of db.bindings) {
      const pat = unit => binding[unit].holes.map(h => ({ x: -h.x, y: h.y, label: h.label }));
      const frontPat = pat('front_unit'), heelPat = pat('heel_unit');
      const fronts = fitUnit(frontPat, existing, tol, maxLateral);
      const heels = fitUnit(heelPat, existing, tol, maxLateral);
      const range = heelRangeOf(binding);
      const fixed = binding.mounting_reference === 'fixed';
      const base = { bindingId: binding.id, bindingName: binding.name, verified: binding.verified !== false };
      const holesOf = (p, f, unit) => p.map(h => ({ x: h.x + f.ref, y: h.y + f.dy, label: h.label, unit }));

      const combos = [];
      for (const f of fronts) for (const h of heels) {
        if (f.used.some(i => h.used.includes(i))) continue;
        if (Math.abs(f.dy - h.dy) > 3) continue;
        const bslEst = fixed ? null : f.ref - h.ref;
        if (fixed) {
          if (Math.abs(f.ref - h.ref) > 2 * tol) continue;
        } else {
          const r = binding.bsl_range_mm || [200, 400];
          if (bslEst < r[0] - range - 5 || bslEst > r[1] + range + 5) continue;
        }
        const nF = frontPat.length, nH = heelPat.length;
        combos.push({
          f, h,
          rms: Math.sqrt((f.rms ** 2 * nF + h.rms ** 2 * nH) / (nF + nH)),
          bslEst,
          mountOffset: (f.ref + h.ref) / 2,
        });
      }
      // Bindings are normally mounted on the mark: when several toe/heel pairings are
      // possible (two old mounts of the same model), prefer the ones centred on it.
      combos.sort((a, b) => Math.abs(a.mountOffset) - Math.abs(b.mountOffset));
      const usedF = new Set(), usedH = new Set();
      for (const c of combos) {
        if (usedF.has(c.f) || usedH.has(c.h)) continue;
        usedF.add(c.f); usedH.add(c.h);
        [...c.f.used, ...c.h.used].forEach(i => explained.add(i));
        full.push({
          ...base, kind: 'full',
          matched: frontPat.length + heelPat.length,
          rms: c.rms, bslEst: c.bslEst, heelRange: range,
          mountOffset: c.mountOffset,
          lateralOffset: (c.f.dy + c.h.dy) / 2,
          holes: [...holesOf(frontPat, c.f, 'front'), ...holesOf(heelPat, c.h, 'heel')],
        });
      }
      for (const f of fronts) if (!usedF.has(f)) {
        partial.push({ ...base, kind: 'toe', matched: frontPat.length, rms: f.rms, ref: f.ref, lateralOffset: f.dy,
          holes: holesOf(frontPat, f, 'front'), key: f.key, used: f.used });
      }
      for (const h of heels) if (!usedH.has(h)) {
        partial.push({ ...base, kind: 'heel', matched: heelPat.length, rms: h.rms, ref: h.ref, lateralOffset: h.dy,
          holes: holesOf(heelPat, h, 'heel'), key: h.key, used: h.used });
      }
    }

    full.sort((a, b) => b.matched - a.matched || a.rms - b.rms);
    // Several models share a toe or heel pattern: group partial matches that explain the same holes.
    const groups = [];
    for (const p of partial.sort((a, b) => b.matched - a.matched || a.rms - b.rms)) {
      // A toe or heel pattern shared with a fully identified binding adds nothing.
      if (p.used.every(i => explained.has(i))) continue;
      const g = groups.find(x => x.kind === p.kind && x.key === p.key);
      if (g) g.alsoFits.push(p.bindingName);
      else groups.push({ ...p, alsoFits: [] });
    }
    return { full: full.slice(0, opts.limit ?? 8), partial: groups.slice(0, opts.limit ?? 8) };
  }

  return {
    DEFAULT_MIN_SEPARATION_MM, REUSE_TOLERANCE_MM, TIGHT_MARGIN_MM, LOOSE_REUSE_MM,
    bslInRange, computeHoles, checkBinding, rankBindings, findClearMountOffset, identify,
  };
});
