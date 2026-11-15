'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../static/matcher.js');
const db = require('../templates/binding_db.json');
const golden = require('./fixtures/matcher_golden.json');

const binding = id => db.bindings.find(b => b.id === id);
const GRIFFON = 'marker-jester-griffon-squire', PIVOT = 'rossignol-fks-look-pivot-2010';
const bare = holes => holes.map(h => ({ x: h.x, y: h.y }));

// Deterministic noise in ±amp
function noise(seed, amp) {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return (s / 2147483647 - 0.5) * 2 * amp; };
}

test('matches the reference implementation on 60 recorded scenarios', () => {
  // Fixtures were produced by the original Python matcher before it was removed.
  for (const [i, c] of golden.cases.entries()) {
    const got = M.rankBindings(db, c.bsl, c.existing, { minSep: c.minSep }).map(r => ({
      id: r.bindingId, new: r.nNew, reusable: r.nReusable, conflicts: r.nConflicts,
      heelOffset: r.heelOffset, score: r.score, status: r.holes.map(h => h.status[0]).join(''),
    }));
    assert.deepEqual(got, c.ranked, `ranking, case ${i}`);

    const m = c.manual;
    const r = M.checkBinding(binding(m.id), c.bsl, c.existing, { minSep: c.minSep, heelOffset: m.heelOffsetIn, mountX: m.mountX });
    assert.equal(r.heelOffset, m.heelOffset, `manual heel offset, case ${i}`);
    assert.equal(r.holes.map(h => h.status[0]).join(''), m.status, `manual statuses, case ${i}`);
    r.holes.forEach((h, k) => {
      assert.ok(Math.abs(h.x - m.holes[k][0]) < 1e-3 && Math.abs(h.y - m.holes[k][1]) < 1e-3, `hole position, case ${i}`);
    });
  }
});

test('every template is laterally symmetric and usable at a mid-range BSL', () => {
  for (const b of db.bindings) {
    const holes = M.computeHoles(b, 310);
    assert.ok(holes.length >= 6, b.id);
    for (const unit of ['front', 'heel']) {
      const ys = holes.filter(h => h.unit === unit).map(h => h.y).sort((a, c) => a - c);
      ys.forEach((y, i) => assert.ok(Math.abs(y + ys[ys.length - 1 - i]) < 1e-9, `${b.id} ${unit} is not symmetric`));
    }
  }
});

test('toe sits in front of the mount mark and heel behind it', () => {
  for (const b of db.bindings) {
    const holes = M.computeHoles(b, 310);
    const mean = unit => { const xs = holes.filter(h => h.unit === unit).map(h => h.x); return xs.reduce((a, c) => a + c, 0) / xs.length; };
    assert.ok(mean('front') > 0 && mean('heel') < 0, b.id);
  }
});

test('BSL outside the binding range is rejected', () => {
  assert.throws(() => M.computeHoles(binding(GRIFFON), 200), RangeError);
  assert.equal(M.rankBindings(db, 200, []).some(r => r.bindingId === GRIFFON), false);
});

test('re-mounting the same binding reuses every hole', () => {
  const existing = bare(M.computeHoles(binding(GRIFFON), 315));
  const r = M.checkBinding(binding(GRIFFON), 315, existing);
  assert.deepEqual([r.nReusable, r.nConflicts, r.heelOffset, r.mountable, r.minClearance], [8, 0, 0, true, null]);
});

test('heel search prefers the smallest shift that clears a conflict', () => {
  const heel = M.computeHoles(binding(GRIFFON), 315).filter(h => h.unit === 'heel');
  // 10 mm beside a heel hole: too far to reuse, too close to keep. Clears from 9.8 mm of shift.
  const r = M.checkBinding(binding(GRIFFON), 315, [{ x: heel[0].x, y: heel[0].y - 10 }]);
  assert.equal(r.nConflicts, 0);
  assert.equal(Math.abs(r.heelOffset), 10);
});

test('a manual heel offset is clamped to the adjustment track', () => {
  const r = M.checkBinding(binding(GRIFFON), 315, [], { heelOffset: 99 });
  assert.equal(r.heelOffset, binding(GRIFFON).heel_unit.adjustment_range_mm);
});

test('"tight" flags a mountable result within 2 mm of the limit', () => {
  const toe = M.computeHoles(binding('rossignol-axial2-look-px'), 315).find(h => h.unit === 'front');
  const at = d => M.checkBinding(binding('rossignol-axial2-look-px'), 315, [{ x: toe.x + d, y: toe.y }]);
  assert.deepEqual([at(13).mountable, at(13).tight], [false, false]);
  assert.deepEqual([at(15).mountable, at(15).tight], [true, true]);
  assert.deepEqual([at(17).mountable, at(17).tight], [true, false]);
  assert.ok(Math.abs(at(15).minClearance - 15) < 1e-9);
});

test('findClearMountOffset returns the nearest conflict-free position', () => {
  const b = binding('rossignol-axial2-look-px');          // no heel adjustment: only the mount can move
  const toe = M.computeHoles(b, 315).find(h => h.unit === 'front');
  const existing = [{ x: toe.x + 8, y: toe.y }];           // 8 mm in front of a toe hole
  assert.equal(M.checkBinding(b, 315, existing).mountable, false);
  const clear = M.findClearMountOffset(b, 315, existing);
  assert.equal(clear.mountable, true);
  assert.equal(clear.mountX, -6);                           // 6 mm back → 14 mm apart
  assert.equal(M.findClearMountOffset(b, 315, existing, { range: 3 }), null);
});

test('identify finds two earlier mounts, their BSL and position', () => {
  const rnd = noise(7, 0.6);
  const existing = [...M.computeHoles(binding(GRIFFON), 315, 0), ...M.computeHoles(binding(PIVOT), 300, 12)]
    .map(h => ({ x: h.x + rnd(), y: h.y + rnd() }));
  const { full, partial } = M.identify(db, existing);
  assert.deepEqual(full.map(m => m.bindingId).sort(), [GRIFFON, PIVOT].sort());
  const g = full.find(m => m.bindingId === GRIFFON), p = full.find(m => m.bindingId === PIVOT);
  assert.ok(Math.abs(g.bslEst - 315) < 1.5 && Math.abs(g.mountOffset) < 1, 'Griffon BSL/position');
  assert.ok(Math.abs(p.bslEst - 300) < 1.5 && Math.abs(p.mountOffset - 12) < 1, 'Pivot BSL/position');
  assert.ok(g.rms < 1 && p.rms < 1);
  // toe patterns shared with other Look models are already explained: no partial noise
  assert.equal(partial.length, 0);
});

test('identify reports a lone toe piece as partial and names look-alike models', () => {
  const toe = bare(M.computeHoles(binding(PIVOT), 310).filter(h => h.unit === 'front'));
  const { full, partial } = M.identify(db, toe);
  assert.equal(full.length, 0);
  assert.equal(partial.length, 1);
  assert.equal(partial[0].kind, 'toe');
  assert.ok(partial[0].alsoFits.length >= 2, 'other Look/Rossignol models share this toe pattern');
});

test('identify stays silent on unrelated holes and on an incomplete unit', () => {
  const scattered = [{ x: 10, y: 3 }, { x: -80, y: -11 }, { x: 140, y: 7 }, { x: -200, y: 2 }, { x: 60, y: -15 }];
  assert.deepEqual(M.identify(db, scattered), { full: [], partial: [] });
  const threeOfFour = bare(M.computeHoles(binding(GRIFFON), 315).filter(h => h.unit === 'front')).slice(0, 3);
  assert.deepEqual(M.identify(db, threeOfFour), { full: [], partial: [] });
});
