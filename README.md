# ⛷️ Ski Binding Conflict Detector

> *"Will my new bindings fit without drilling new holes?" — finally, a tool that attempts to answer this.*

A web application that helps you figure out whether a binding can be mounted on a ski that already has holes, without creating conflicts with existing drill points.

You photograph the ski with a measuring tape lying on it, calibrate the image geometry, mark the existing holes by hand, and the app checks every binding in the database against your hole pattern — telling you what's mountable as-is, what needs new holes only, and what conflicts.

Everything runs in your browser. The photo is never uploaded: there is no server-side code at all, just static files.

**⚠️ Alpha software. Use at your own risk. Always verify results with a certified ski technician before drilling anything.**

---

## What it does

1. **Calibrate** — you click 2–4 marks on a measuring tape laid on the ski. With 3+ marks the app corrects for camera parallax (perspective distortion along the ski axis). You also mark the ski's centre mount reference and optionally draw the ski axis for higher accuracy.
2. **Mark holes** — click on the photo to mark every existing drill hole. No auto-detection; manual placement is deliberate (see [Removed features](#removed-for-now)).
3. **Analyze** — the app projects every binding template onto the ski coordinate system and classifies each hole as reusable, new, or conflicting. Results are sorted by compatibility, with the clearance in millimetres rather than just a verdict.
4. **Explore** — click any binding row to overlay its template on the photo. Sliders adjust BSL, heel position, and mounting point offset instantly.
5. **Identify** — one click works out which binding made the existing holes, for what boot sole length, and where it sat. A clean match doubles as proof that your calibration is right.
6. **Verify** — print a 1:1 check sheet, lay it on the ski, and see whether the printed holes sit on the real ones before you drill anything.

---

## Screenshots

![Calibration](https://skipass.fr/p/imagebank/2/4/1/241806-bxKTu3zX.pn "Calibration")
![Holes marking](https://skipass.fr/p/imagebank/2/4/1/241807-4183A1Rd.png "Holes marking")
![Binding analysis](https://skipass.fr/p/imagebank/2/4/1/241808-pz0Hb4w8.png "Binding analysis")
---

## Features

- 🔒 **Nothing leaves your device** — the photo is processed in the browser; the site is static files only.
- 📏 **Parallax correction** — 3 or 4 tape marks fit a 1-D projective transform along the ski axis. Lengthwise it is exact for a camera tilted along the ski; the sideways scale is derived from the same fit.
- 🗺️ **Axis calibration** — optional edge-to-edge transversal lines constrain the ski axis more precisely than the tape alone; the mounting point is projected onto the computed centreline.
- 🖊️ **Manual hole editor** — click to add, undo or tap a mark to remove. Simple and reliable.
- 🎿 **13 binding templates** — alpine and touring/LT; see full list below.
- 🦶 **Heel-track search** — for each binding the heel piece is tried along its whole adjustment range, so a conflict that goes away by sliding the heel 1 cm is reported as mountable, with the offset. If that is not enough, the app tells you the smallest mount-point shift that clears it.
- 🔍 **Interactive overlay** — BSL, heel offset, and mount offset sliders update the overlay instantly; a second cross shows the offset mounting point without moving the calibration reference.
- 🕵️ **Previous-binding identification** — pure geometry: every template's toe and heel pattern is fitted to the existing holes.
- 🖨️ **1:1 check sheet** — existing and planned holes at true size, one sheet per binding unit, A4 or Letter.
- 📐 **Debug grid** — toggle a 1 cm or ½ cm grid aligned to the ski axis to verify calibration visually: the grid lines should fall on the tape's centimetre marks.
- 🐳 **Docker deployment** — one `docker compose up --build` and you're done. Or skip Docker: any static file host works.

---

## Quick Start (Docker)

```bash
git clone https://github.com/derdide/solemate.git
cd solemate

docker compose up --build -d

# Open http://localhost:8000
```

No API keys or environment variables required. The container is a small static file server (Caddy) and nothing else.

### Without Docker

The app is `index.html`, `static/` and `templates/`. Copy those three anywhere that serves files — an existing Caddy or nginx (`root` + `file_server`), GitHub Pages, Cloudflare Pages. It needs http(s); opening `index.html` straight from disk does not work because the browser will not let it load the template database.

---

## Local Development

```bash
# any static server will do
python3 -m http.server 8765 --bind 127.0.0.1

# Open http://localhost:8765

# Tests (Node 20+, no dependencies to install)
npm test
```

| File | What it holds |
|------|---------------|
| `static/calib.js` | photo pixels ↔ ski millimetres, parallax correction |
| `static/matcher.js` | templates vs. existing holes: conflicts, heel search, identification |
| `static/print.js` | the 1:1 check sheets |
| `static/app.js` | the UI |

The first three are pure functions with no DOM access; `tests/` exercises them, including 60 recorded scenarios from the original Python implementation.

---

## Hosting it safely

There is no server-side code: the server never sees a photo, takes no input, and keeps no state. What is left to get right is ordinary static hosting.

- **The container is locked down** in `docker-compose.yml`: unprivileged user, read-only filesystem, no Linux capabilities, tight memory/CPU/PID limits, and every request that is not a `GET` is refused. Keep those settings if you write your own compose file.
- **Don't expose port 8000 directly to the internet.** Put a reverse proxy with TLS in front. If the proxy runs on the same host, publish the port as `127.0.0.1:8000:8000`; Docker-published ports are not filtered by ufw/firewalld.
- **Or don't run a container at all.** Pointing your existing web server at the three paths above removes the extra moving part, and hosting them on GitHub Pages or Cloudflare Pages takes your home network out of the picture entirely.
- **If your origin sits behind a CDN proxy** (e.g. Cloudflare), make sure it cannot be reached around the proxy: use a tunnel, or only accept the CDN's address ranges on the forwarded port.

---

## How to Use It

### Step 1 — Choose a photo & calibrate

Take a photo of your ski laid flat with a measuring tape lying **on the topsheet**, parallel to the ski. Hold the camera square above the binding area. Phone photos work fine; uniform lighting helps.

Where the tape sits matters more than anything else: it has to be in the same plane as the holes. A tape on the bench next to the ski is ~15 mm further from the camera, which alone shifts every measurement by 2–4 mm (see [Accuracy](#accuracy)).

Click the calibration tools in order:

| Button | What to do |
|--------|------------|
| **① Tape** | Click 2–4 marks at equal known spacing (e.g. every 100 mm, heel → tip). 3+ marks enable parallax correction. |
| **② Mount pt** | Click the ski's centre mark — the half-sole reference line printed on the ski. |
| **③ Axis L1** *(optional)* | Click one ski edge, then the opposite edge at the heel zone. Defines the ski axis better than the tape alone. |
| **④ Axis L2** *(optional)* | Same as L1 but at the tip zone. Together L1+L2 give a precise centreline. |
| **↩ Flip axis** | If the orange **→TIP** arrow points the wrong way, click this. |

Click **Calibrate →**. The app warns you if only two marks were given, if the perspective is strong, or if the marks do not agree with each other.

Zoom with the wheel or a pinch; pan by dragging (two fingers on touch). On touch screens a mark lands slightly above your fingertip so you can see where it goes.

### Step 2 — Mark Holes

Click on the photo to mark each existing drill hole. To remove a mark, tap its centre or use **Undo**. There is no auto-detection. Going back to recalibrate keeps your marks.

Set your **Boot sole length (BSL)**, binding **category**, and **minimum hole separation** (default 14 mm), then click **Analyze →**.

### Step 3 — Results

The results table lists every binding sorted by compatibility:

- **OK** — all template holes either reuse existing holes or land in undamaged areas
- **TIGHT** — mountable, but the closest new hole is within 2 mm of your limit, which is less than a photo can reliably measure
- **CONFLICT** — one or more template holes fall too close to an existing hole

The **Clear.** column is the smallest distance between a hole you would drill and an existing one. The **Heel** column is where the app placed the heel piece on its adjustment track to avoid conflicts.

Click any row to overlay the binding template on your ski photo. The sliders let you adjust:

- **BSL** — boot sole length (changes toe/heel separation)
- **Heel offset** — if the binding's heel unit has a longitudinal adjustment range. On **auto** the app picks the best position; move the slider to take over.
- **Mount offset** — shift the entire binding forward/back relative to your calibration point

For a conflicting binding the app also reports the nearest conflict-free mount position within ±30 mm, if there is one.

The **yellow cross** always marks your original calibration point. When mount offset ≠ 0 a **white cross** appears at the effective mounting position.

The **debug grid** (1 cm or ½ cm) overlays a coordinate grid aligned to the ski axis, centred on the mount point — useful for verifying that calibration and parallax correction look right.

### What was mounted before?

**Identify** fits every template to the holes you marked. A full match needs every hole of both the toe and the heel piece to coincide with an existing hole; it then tells you the binding, the boot sole length it was mounted for (to within the heel's adjustment range), and how far from your mount mark it was centred. Several old mounts on the same ski are found separately.

It is also the best check of your calibration you can get for free: a jig-drilled pattern is far more precise than clicks on a tape, so if the holes fit a template to a few tenths of a millimetre, your scale and axis are right. If the old binding is not among the templates, nothing is reported.

### Print a 1:1 check sheet

**Print 1:1** produces one landscape sheet per binding unit with the existing holes (grey) and the planned holes at true size. Print at 100 % / "actual size" and measure the 100 mm bar on the sheet. Lay the sheet on the ski with the centreline and the mount line aligned: the grey circles must sit on the real holes. If they do not, the calibration is off and the result should not be trusted.

It is a check sheet, not a drilling jig.

---

## Accuracy

Measured by driving the app with perfect clicks on synthetic photos of a known hole pattern (camera 1.2 m above the ski, holes up to 215 mm from the mount mark). Real clicks add their own error on top.

| Situation | Worst hole error |
|-----------|------------------|
| Camera square-on, tape on the topsheet | 0.0 mm |
| Camera tilted 10° along the ski, 3–4 tape marks | 0.3 mm |
| Camera tilted 20° along the ski, 3–4 tape marks | 1.2 mm |
| Camera tilted 10° along the ski, only 2 tape marks | 6.4 mm |
| Tape 15 mm below the topsheet (on the bench beside the ski) | 2.7 mm |
| Camera tilted 10° sideways | 1.7 mm |

So: tape on the ski, three or more marks, camera as square as you can manage — and treat anything within a couple of millimetres of your limit as undecided until the printed sheet says otherwise.

---

## Binding Database

Source PDFs exist for ~33 bindings. Of those, **13 have been manually transcribed and confirmed** — dimensions cross-checked against the PDF and physically verified on bindings I actually own. The remaining ~20 have source PDFs but have not been entered into the database yet; contributions are welcome.

*"Confirmed" here means: I worked out the hole coordinates from the PDF myself, and I own the binding, so I could check it. It does not mean I physically drilled a ski and measured the result. Use your own judgement.*

### In the database & confirmed (13)

#### Alpine (12)

| Binding |
|---------|
| Marker Jester / Griffon / Squire |
| Salomon Alpine (4-hole toe) |
| Salomon Alpine (3-hole toe) |
| Salomon STH2 |
| Salomon Warden MNC |
| Salomon Rental |
| Tyrolia / Head Alpine |
| Tyrolia / Head AAAttack13 Demo/Rental |
| Rossignol Axial1 / Look Pivot (pre-2005) |
| Rossignol Axial2 / Look PX |
| Rossignol FKS / Look Pivot (post-2010) |
| Rossignol Axial Racing / Look PX Racing |

#### Touring / LT (1)

| Binding |
|---------|
| Salomon Shift MNC 13 |

### Source PDFs available but not yet entered (~20)

These bindings have manufacturer drill templates on file. What's missing is someone working out the hole coordinates and entering them in `binding_db.json`. If you own one of these and want to contribute, this is the highest-value thing you can do.

**Particularly welcome:** Marker Duke / Baron / Tour F10/F12 — both EPF and non-EPF variants. The EPF toe piece uses a different hole pattern and it is not yet clear how best to model the two variants together in the database schema. If you have both and can measure them, please open an issue or PR.

Other bindings with PDFs on file include (incomplete list): Marker Duke EPF, Marker M-Series, Marker Ten Free, Salomon Guardian / Atomic Tracker, Naxo NX01/NX21 (alpine and touring), Tyrolia/Head AAAmbition, Tyrolia/Head AAAmbition Carbon, Dynafit Radical V1 and V2, Dynafit TLT Superlite, Dynafit TLT Vertical/Speed / G3 Onyx, Dynafit Beast 14 and 16, G3 Ion / Zed, Fritschi Diamir Freeride, Fritschi Diamir FreeridePro + Eagle, Fritschi Diamir Vipec, Salomon MTN, Marker Kingpin.

### Adding bindings

Templates live in `templates/binding_db.json`. Since the templates directory is mounted as a Docker volume you can add or edit bindings without rebuilding the container; reload the page to pick up the change.

Coordinates follow the paper drill templates: hole `x` grows toward the **tail**, `y` toward the skier's right. For `"mounting_reference": "half_bsl"` the front unit's `x` is measured from the boot tip and the heel unit's from the boot heel; for `"fixed"` (rental/demo plates) both are measured from the mount point. `heel_unit.adjustment_range_mm` is how far the heel may slide either way. `npm test` checks every template for symmetry and for the toe being in front of the heel, which catches a flipped sign.

---

## Removed for Now

These features existed in earlier versions and were deliberately removed. They may return.

| Feature | Status | Reason |
|---------|--------|--------|
| **Auto hole detection** (OpenCV HoughCircles / blob) | Removed | Results were unreliable — too many false positives on dark skis or complex graphics. Manual marking is slower but actually correct. |
| **AI hole detection** (Ollama / Claude Vision) | Removed | Code existed but was largely untested in practice. Removed rather than ship something misleading. |
| **Multi-BSL range probing** (exact BSL ±30 mm) | Removed | Its job — finding a nearby position without conflicts — is done directly: the heel piece is searched along its adjustment track, and the nearest conflict-free mount offset is reported. Both are positions you can actually choose; a different boot sole length is not. |
| **Access token auth** | Removed | Nothing to protect: the site is static and receives no data. |
| **Server-side processing** (FastAPI, OpenCV) | Removed | Everything moved into the browser. The server no longer parses images, so there is no upload to abuse and no photo ever leaves the device. |

---

## Limitations

- **Only 13 bindings are active.** Source PDFs exist for ~33, but the remaining ~20 have not been transcribed yet. The database only includes bindings whose hole coordinates have been worked out and confirmed — see the [Binding Database](#binding-database) section for how to contribute.
- **Parallax correction assumes the camera is only tilted along the ski.** Lengthwise it is then exact; sideways it is approximate (about 0.3 mm at 10°, 1.2 mm at 20°). A camera tilted sideways, or a tape that is not in the plane of the topsheet, is not corrected at all — see [Accuracy](#accuracy).
- **"Reusable" means within 2 mm on the photo.** That cannot distinguish a hole that is exactly where the template wants it from one that is 1.5 mm off — and the second cannot be reused. The app flags these; the printed sheet settles it.
- **No warranty whatsoever.** Drilling ski holes in the wrong place can damage the ski, compromise binding retention, and create a safety hazard. This tool is a planning aid, not a substitute for a qualified ski technician with a calibrated drill jig.

---

## Project Status

**Alpha. Personal project.** Built because I needed it; shared because someone else might too.

Development continues when time allows. No roadmap, no support SLA. Bug reports and binding template PRs are welcome.

---

## Contributing

- **New binding templates** — most useful contribution. Include the source PDF, note whether you verified it on a real ski, and open a PR.
- **Bug reports** — include a photo (or describe it), what you expected, and what happened.
- **Code** — keep it simple. No build step, no dependencies; run `npm test` before opening a PR.

---

## License

MIT License — do whatever you want with it, keep the copyright notice.

```
Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*Built with plain JavaScript and manufacturer PDF drill templates.*
