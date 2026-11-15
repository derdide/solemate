/* print.js — 1:1 paper check sheets.
 *
 * Builds one SVG page per binding unit (toe piece, heel piece) in real millimetres.
 * Each page shows the holes the app believes already exist and the holes the binding
 * needs, so the sheet can be laid on the ski before drilling: if the printed "existing"
 * circles do not sit on the real holes, the calibration is off and the verdict is void.
 *
 * Pure string building, no DOM: runs in the browser (window.PrintSheet) and under Node.
 * Ski frame: x toward the tip, y toward the skier's right. On paper the tip points right
 * and the sheet is read from above, so +y is drawn downward.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PrintSheet = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PAPER = {
    a4:     { w: 297,   h: 210,   name: 'A4' },
    letter: { w: 279.4, h: 215.9, name: 'Letter' },
  };
  const MARGIN = 12;       // unprintable border assumed on each side, mm
  const PAD = 18;          // paper kept around the outermost hole, mm
  const HOLE_R = 2.05;     // 4.1 mm drill

  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f = n => (Math.round(n * 100) / 100).toString();
  const signed = n => (n > 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n).toFixed(1);

  const STATUS = {
    reusable: { colour: '#15803d', word: 'reuse' },
    new:      { colour: '#c2410c', word: 'drill' },
    conflict: { colour: '#b91c1c', word: 'CONFLICT' },
  };

  /**
   * Choose the stretch of ski a page covers and the line used to position it lengthwise.
   * Preferably that line is the ski's own mount mark (x = 0). When the unit is too far
   * from it to fit on one sheet, a reference line at a round distance is used instead.
   */
  function layoutUnit(xs, usable) {
    let min = Math.min(...xs) - PAD, max = Math.max(...xs) + PAD;
    if (Math.max(max, 6) - Math.min(min, -6) <= usable) {
      return { min: Math.min(min, -6), max: Math.max(max, 6), refX: 0 };
    }
    // Round distance (multiple of 50 mm) on the mount-mark side of the unit.
    const towardMark = min > 0 ? -1 : 1;                      // direction from the unit to x = 0
    const edge = towardMark < 0 ? min : max;                  // unit edge facing the mark
    let refX = towardMark < 0 ? Math.floor(edge / 50) * 50 : Math.ceil(edge / 50) * 50;
    if (towardMark < 0) min = Math.min(min, refX - 6); else max = Math.max(max, refX + 6);
    if (max - min > usable) {                                 // still too long: put the line inside the unit's padding
      refX = towardMark < 0 ? Math.ceil((Math.min(...xs) - PAD) / 10) * 10 : Math.floor((Math.max(...xs) + PAD) / 10) * 10;
      min = Math.min(...xs) - PAD; max = Math.max(...xs) + PAD;
    }
    return { min, max, refX };
  }

  function refLabel(refX) {
    if (refX === 0) return 'SKI MOUNT MARK — align with the mark on the ski';
    const d = Math.abs(refX);
    return `REFERENCE LINE — ${d} mm ${refX > 0 ? 'in front of' : 'behind'} the ski's mount mark`;
  }

  function page(o, unit, unitName, paper) {
    const holes = o.holes.filter(h => h.unit === unit);
    if (!holes.length) return null;
    const lay = layoutUnit(holes.map(h => h.x), paper.w - 2 * MARGIN);
    const span = lay.max - lay.min;
    const ox = (paper.w - span) / 2 - lay.min;      // paper X of ski x = 0
    const cy = paper.h / 2 - 4;                     // paper Y of the centreline
    const X = x => f(ox + x), Y = y => f(cy + y);
    const half = 58;                                // half-height of the cross lines
    const out = [];

    out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${f(paper.w)} ${f(paper.h)}" width="${f(paper.w)}mm" height="${f(paper.h)}mm" font-family="Helvetica, Arial, sans-serif" fill="none" stroke-linecap="round">`);
    out.push(`<rect x="0" y="0" width="${f(paper.w)}" height="${f(paper.h)}" fill="#fff"/>`);

    // Header
    out.push(`<g fill="#000" stroke="none">`);
    out.push(`<text x="${MARGIN}" y="${MARGIN + 5}" font-size="5" font-weight="700">${esc(o.bindingName)} — ${unitName}</text>`);
    out.push(`<text x="${MARGIN}" y="${MARGIN + 10.5}" font-size="3.2">Boot sole ${f(o.bslMm)} mm · heel ${signed(o.heelOffset)} mm · mount offset ${signed(o.mountX)} mm · min. separation ${f(o.minSep)} mm · ${esc(o.date)}</text>`);
    out.push(`<text x="${f(paper.w - MARGIN)}" y="${MARGIN + 5}" font-size="5" font-weight="700" text-anchor="end">TIP →</text>`);
    out.push(`<text x="${f(paper.w - MARGIN)}" y="${MARGIN + 10.5}" font-size="3.2" text-anchor="end">seen from above · sheet ${o.pageNo} of ${o.pageCount}</text>`);
    out.push(`</g>`);

    // Centreline
    out.push(`<line x1="${MARGIN}" y1="${Y(0)}" x2="${f(paper.w - MARGIN)}" y2="${Y(0)}" stroke="#000" stroke-width="0.2" stroke-dasharray="8 2 1 2"/>`);
    out.push(`<text x="${f(paper.w - MARGIN)}" y="${f(cy - 1.5)}" font-size="2.8" fill="#000" text-anchor="end">ski centreline</text>`);

    // Positioning line
    out.push(`<line x1="${X(lay.refX)}" y1="${f(cy - half)}" x2="${X(lay.refX)}" y2="${f(cy + half)}" stroke="#000" stroke-width="0.35"/>`);
    const [refTitle, refNote] = refLabel(lay.refX).split(' — ');
    out.push(`<text transform="translate(${f(ox + lay.refX - 1.5)} ${f(cy + half)}) rotate(-90)" font-size="2.8" font-weight="700" fill="#000">${esc(refTitle)}</text>`);
    out.push(`<text transform="translate(${f(ox + lay.refX - 1.5)} ${f(cy - 4)}) rotate(-90)" font-size="2.6" fill="#000">${esc(refNote)}</text>`);

    // Boot centre, when the binding is not mounted on the mark
    if (Math.abs(o.mountX) >= 0.05 && o.mountX > lay.min && o.mountX < lay.max && lay.refX === 0) {
      out.push(`<line x1="${X(o.mountX)}" y1="${f(cy - half + 14)}" x2="${X(o.mountX)}" y2="${f(cy + half - 14)}" stroke="#000" stroke-width="0.25" stroke-dasharray="2 1.5"/>`);
      out.push(`<text transform="translate(${f(ox + o.mountX + 3.6)} ${f(cy + half - 14)}) rotate(-90)" font-size="2.6" fill="#000">boot centre (${signed(o.mountX)} mm)</text>`);
    }

    // Existing holes, as measured from the photo
    for (const e of o.existing) {
      if (e.x < lay.min - HOLE_R || e.x > lay.max + HOLE_R) continue;
      out.push(`<circle cx="${X(e.x)}" cy="${Y(e.y)}" r="${HOLE_R}" stroke="#000" stroke-width="0.3" fill="#d4d4d4"/>`);
    }

    // Binding holes
    for (const h of holes) {
      const st = STATUS[h.status] || STATUS.new;
      const cx = ox + h.x, yy = cy + h.y, arm = 5.5;
      out.push(`<g stroke="${st.colour}" stroke-width="0.3">`);
      out.push(`<circle cx="${f(cx)}" cy="${f(yy)}" r="${HOLE_R}"${h.status === 'reusable' ? ' stroke-width="0.7"' : ''}/>`);
      out.push(`<line x1="${f(cx - arm)}" y1="${f(yy)}" x2="${f(cx + arm)}" y2="${f(yy)}"/>`);
      out.push(`<line x1="${f(cx)}" y1="${f(yy - arm)}" x2="${f(cx)}" y2="${f(yy + arm)}"/>`);
      if (h.status === 'conflict') {
        out.push(`<line x1="${f(cx - 3.5)}" y1="${f(yy - 3.5)}" x2="${f(cx + 3.5)}" y2="${f(yy + 3.5)}" stroke-width="0.6"/>`);
        out.push(`<line x1="${f(cx + 3.5)}" y1="${f(yy - 3.5)}" x2="${f(cx - 3.5)}" y2="${f(yy + 3.5)}" stroke-width="0.6"/>`);
      }
      const below = h.y >= 0;
      out.push(`<text x="${f(cx)}" y="${f(yy + (below ? arm + 3.6 : -arm - 1.4))}" font-size="2.8" fill="${st.colour}" stroke="none" text-anchor="middle">${st.word}</text>`);
      out.push(`</g>`);
    }

    // Scale check: both directions, because some drivers stretch one axis
    const by = paper.h - MARGIN - 13, bx = MARGIN;
    out.push(`<g stroke="#000" stroke-width="0.25">`);
    out.push(`<line x1="${bx}" y1="${f(by)}" x2="${bx + 100}" y2="${f(by)}"/>`);
    for (let i = 0; i <= 10; i++) out.push(`<line x1="${bx + i * 10}" y1="${f(by - (i % 5 ? 1.5 : 3))}" x2="${bx + i * 10}" y2="${f(by)}"/>`);
    out.push(`<line x1="${f(paper.w - MARGIN)}" y1="${f(by)}" x2="${f(paper.w - MARGIN)}" y2="${f(by - 50)}"/>`);
    for (let i = 0; i <= 5; i++) out.push(`<line x1="${f(paper.w - MARGIN - (i % 5 ? 1.5 : 3))}" y1="${f(by - i * 10)}" x2="${f(paper.w - MARGIN)}" y2="${f(by - i * 10)}"/>`);
    out.push(`</g>`);
    out.push(`<g fill="#000" stroke="none" font-size="2.8">`);
    out.push(`<text x="${bx}" y="${f(by + 4)}" font-weight="700">This bar must measure exactly 100 mm (the one on the right 50 mm). If not, reprint at 100 % / "actual size".</text>`);
    out.push(`<text x="${bx}" y="${f(by + 8)}">Lay the sheet on the ski: centreline on the ski's centre, vertical line as labelled. The grey circles must sit on the real holes.</text>`);
    out.push(`<text x="${bx}" y="${f(by + 12)}">If they do not, the photo calibration is off and this result cannot be trusted. A check sheet, not a drilling jig.</text>`);
    out.push(`</g>`);

    // Legend
    const lx = paper.w - MARGIN - 64, ly = MARGIN + 17;
    out.push(`<g font-size="2.8" stroke="none" fill="#000">`);
    out.push(`<circle cx="${f(lx)}" cy="${f(ly)}" r="${HOLE_R}" stroke="#000" stroke-width="0.3" fill="#d4d4d4"/><text x="${f(lx + 4)}" y="${f(ly + 1)}">existing hole (as measured)</text>`);
    let row = 1;
    for (const key of ['reusable', 'new', 'conflict']) {
      if (!holes.some(h => h.status === key)) continue;
      const st = STATUS[key], y = ly + row * 5.5;
      out.push(`<circle cx="${f(lx)}" cy="${f(y)}" r="${HOLE_R}" stroke="${st.colour}" stroke-width="${key === 'reusable' ? 0.7 : 0.3}" fill="none"/>`);
      out.push(`<text x="${f(lx + 4)}" y="${f(y + 1)}" fill="${st.colour}">${{ reusable: 'binding hole on an existing hole', new: 'binding hole to drill', conflict: 'binding hole too close to an existing hole' }[key]}</text>`);
      row++;
    }
    out.push(`</g>`);

    out.push(`</svg>`);
    return out.join('');
  }

  /**
   * @param {object} o
   *   bindingName, bslMm, heelOffset, mountX, minSep, date,
   *   holes:    [{x, y, unit:'front'|'heel', status}]   binding holes (ski mm)
   *   existing: [{x, y}]                                 existing holes (ski mm)
   *   paper:    'a4' | 'letter'
   * @returns {{ paper:{w,h,name}, pages:string[] }}  one landscape SVG per unit
   */
  function build(o) {
    const paper = PAPER[o.paper] || PAPER.a4;
    const units = [['front', 'toe piece'], ['heel', 'heel piece']].filter(([u]) => o.holes.some(h => h.unit === u));
    const pages = units.map(([u, name], i) => page({ ...o, pageNo: i + 1, pageCount: units.length }, u, name, paper));
    return { paper, pages };
  }

  return { build, layoutUnit, PAPER };
});
