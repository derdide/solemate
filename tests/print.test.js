'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const P = require('../static/print.js');
const M = require('../static/matcher.js');
const db = require('../templates/binding_db.json');

function sheet(id, bsl, paper = 'a4', extra = {}) {
  const b = db.bindings.find(x => x.id === id);
  const existing = [{ x: 150, y: 18 }, { x: -140, y: -16 }];
  const r = M.checkBinding(b, bsl, existing);
  return P.build({ bindingName: b.name, bslMm: bsl, heelOffset: r.heelOffset, mountX: 0, minSep: 14, date: '2026-01-01',
    holes: r.holes, existing, paper, ...extra });
}

test('one landscape page per unit, at true size', () => {
  const s = sheet('marker-jester-griffon-squire', 315);
  assert.equal(s.pages.length, 2);
  for (const svg of s.pages) {
    assert.match(svg, /viewBox="0 0 297 210" width="297mm" height="210mm"/);   // 1 user unit = 1 mm
    assert.match(svg, /SKI MOUNT MARK/);
    assert.equal((svg.match(/<svg/g) || []).length, 1);
  }
  assert.match(sheet('marker-jester-griffon-squire', 315, 'letter').pages[0], /width="279.4mm" height="215.9mm"/);
});

test('every binding fits on the sheet at both ends of its BSL range', () => {
  for (const b of db.bindings) {
    for (const bsl of b.bsl_range_mm || [310]) {
      for (const paper of ['a4', 'letter']) {
        const holes = M.computeHoles(b, bsl);
        for (const unit of ['front', 'heel']) {
          const xs = holes.filter(h => h.unit === unit).map(h => h.x);
          const usable = P.PAPER[paper].w - 24;
          const lay = P.layoutUnit(xs, usable);
          assert.ok(lay.max - lay.min <= usable + 1e-9, `${b.id} ${unit} @${bsl} on ${paper}: ${lay.max - lay.min} mm`);
          assert.ok(lay.min <= Math.min(...xs) - 5 && lay.max >= Math.max(...xs) + 5, `${b.id} ${unit}: holes too close to the paper edge`);
          assert.ok(lay.refX >= lay.min && lay.refX <= lay.max, `${b.id} ${unit}: positioning line off the sheet`);
        }
      }
    }
  }
});

test('a unit too far from the mount mark gets a reference line at a stated distance', () => {
  const lay = P.layoutUnit([-154, -259], 273);
  assert.equal(lay.refX, -100);
  const svg = sheet('rossignol-axial2-look-px', 360).pages[1];
  assert.match(svg, /REFERENCE LINE/);
  assert.match(svg, /100 mm behind the ski&#39;s mount mark/);
});

test('binding names are escaped', () => {
  const svg = sheet('marker-jester-griffon-squire', 315, 'a4', { bindingName: '<script>x</script> & co' }).pages[0];
  assert.equal(svg.includes('<script>'), false);
  assert.match(svg, /&lt;script&gt;x&lt;\/script&gt; &amp; co/);
});
