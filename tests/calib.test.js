'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const C = require('../static/calib.js');

// A pinhole camera 1.2 m above the ski, looking down; tip to the right in the photo.
const W = 4000, H = 3000, F = 3000;
function camera({ pitch = 0, roll = 0, height = 1200 } = {}) {
  const rad = d => d * Math.PI / 180;
  const cp = Math.cos(rad(pitch)), sp = Math.sin(rad(pitch)), cr = Math.cos(rad(roll)), sr = Math.sin(rad(roll));
  const R = v => {                       // Ry(pitch) · Rx(roll)
    const [x, y, z] = [v[0], cr * v[1] - sr * v[2], sr * v[1] + cr * v[2]];
    return [cp * x + sp * z, y, -sp * x + cp * z];
  };
  const ax = R([1, 0, 0]), down = R([0, -1, 0]), az = R([0, 0, -1]);
  const cam = az.map(v => -v * height);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  // (x toward tip, y toward skier's right, z up) → pixel
  return (x, y, z = 0) => {
    const d = [x - cam[0], -y - cam[1], z - cam[2]];
    return { x: F * dot(d, ax) / dot(d, az) + W / 2, y: F * dot(d, down) / dot(d, az) + H / 2 };
  };
}

const HOLES = [[178.5, 18], [178.5, -18], [147.5, 18], [147.5, -18], [-132.5, 16], [-132.5, -16], [-212.5, 16], [-212.5, -16]];

function calibrate(proj, { marks = [-150, -50, 50, 150], tapeY = 40, tapeZ = 0 } = {}) {
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const a = mid(proj(-380, 45), proj(-380, -45)), b = mid(proj(380, 45), proj(380, -45));
  const len = Math.hypot(b.x - a.x, b.y - a.y);
  const axis = { dx: (b.x - a.x) / len, dy: (b.y - a.y) / len };
  const m = proj(0, 0), t = (m.x - a.x) * axis.dx + (m.y - a.y) * axis.dy;
  return C.build({
    tape: marks.map(x => proj(x, tapeY, tapeZ)), spacingMm: marks[1] - marks[0],
    mount: { x: a.x + t * axis.dx, y: a.y + t * axis.dy }, axis, imageWidth: W, imageHeight: H,
  });
}

function maxError(cal, proj) {
  let along = 0, lateral = 0;
  for (const [x, y] of HOLES) {
    const mm = C.toMm(cal, proj(x, y));
    along = Math.max(along, Math.abs(mm.x - x));
    lateral = Math.max(lateral, Math.abs(mm.y - y));
  }
  return { along, lateral };
}

test('square-on photo is measured exactly, with the correct handedness', () => {
  const proj = camera();
  const cal = calibrate(proj);
  const e = maxError(cal, proj);
  assert.ok(e.along < 0.01 && e.lateral < 0.01, JSON.stringify(e));
  // tip to the right in the photo → the skier's right is the lower half of the image
  assert.ok(C.toMm(cal, { x: cal.mount.x, y: cal.mount.y + 50 }).y > 0);
  assert.ok(Math.abs(cal.mmPerPx - 0.4) < 1e-6);
  assert.ok(cal.scaleSpread < 1e-6);
});

test('camera tilted along the ski: lengthwise exact, sideways within a fraction of a millimetre', () => {
  for (const [pitch, lateralLimit] of [[10, 0.35], [20, 1.25]]) {
    const proj = camera({ pitch });
    const e = maxError(calibrate(proj), proj);
    assert.ok(e.along < 0.05, `pitch ${pitch}: along ${e.along}`);
    assert.ok(e.lateral < lateralLimit, `pitch ${pitch}: lateral ${e.lateral}`);
  }
});

test('three marks are enough for the correction; two are not, and say so', () => {
  const proj = camera({ pitch: 10 });
  const three = calibrate(proj, { marks: [-100, 0, 100] });
  assert.equal(three.corrected, true);
  assert.ok(maxError(three, proj).along < 0.05);
  const two = calibrate(proj, { marks: [-50, 50] });
  assert.equal(two.corrected, false);
  assert.ok(maxError(two, proj).along > 3, 'uncorrected perspective costs millimetres');
});

test('known limits: tape below the topsheet and sideways tilt are not corrected', () => {
  // Documented in the README; these numbers are why the UI asks for the tape on the ski.
  const flat = camera();
  assert.ok(maxError(calibrate(flat, { tapeZ: -15 }), flat).along > 2);
  const rolled = camera({ roll: 10 });
  assert.ok(maxError(calibrate(rolled), rolled).along > 1);
});

test('toPx inverts toMm, and strong perspective is reported', () => {
  const proj = camera({ pitch: 20 });
  const cal = calibrate(proj);
  for (const [x, y] of HOLES) {
    const px = C.toPx(cal, { x, y }), back = C.toMm(cal, px);
    assert.ok(Math.abs(back.x - x) < 1e-6 && Math.abs(back.y - y) < 1e-6);
  }
  assert.ok(cal.scaleSpread > 0.12);
  assert.ok(Math.abs(C.pxPerMm(cal, 0) - 1 / cal.mmPerPx) / (1 / cal.mmPerPx) < 0.1);
});

test('a misplaced tape mark shows up as a fit residual', () => {
  const proj = camera({ pitch: 10 });
  const tape = [-150, -50, 50, 150].map(x => proj(x, 40));
  tape[2].x += 12;                                         // ≈ 5 mm off
  const cal = C.build({ tape, spacingMm: 100, mount: proj(0, 0), axis: { dx: 1, dy: 0 }, imageWidth: W, imageHeight: H });
  assert.ok(cal.residualMm > 0.7, String(cal.residualMm));
});

test('bad input is rejected with a readable message', () => {
  const base = { spacingMm: 100, mount: { x: 500, y: 500 }, axis: { dx: 1, dy: 0 }, imageWidth: W, imageHeight: H };
  assert.throws(() => C.build({ ...base, tape: [{ x: 1, y: 1 }] }), /at least 2/);
  assert.throws(() => C.build({ ...base, tape: [{ x: 100, y: 1 }, { x: 100.2, y: 9 }] }), /coincide/);
  assert.throws(() => C.build({ ...base, tape: [{ x: 1, y: 1 }, { x: 300, y: 1 }], axis: { dx: 0, dy: 0 } }), /axis/);
  assert.throws(() => C.build({ ...base, tape: [{ x: 1, y: 1 }, { x: 300, y: 1 }], spacingMm: 0 }), /spacing/);
  // wildly inconsistent spacing puts the perspective pole inside the photo
  assert.throws(() => C.build({ ...base, tape: [{ x: 100, y: 1 }, { x: 110, y: 1 }, { x: 2000, y: 1 }] }), /inconsistent/);
});
