# ⛷️ Ski Binding Conflict Detector

> *"Will my new bindings fit without drilling new holes?" — finally, a tool that attempts to answer this.*

A web application that helps you figure out whether a binding can be mounted on a ski that already has holes, without creating conflicts with existing drill points.

You photograph the ski alongside a measuring tape, calibrate the image geometry, mark the existing holes by hand, and the app checks every binding in the database against your hole pattern — telling you what's mountable as-is, what needs new holes only, and what conflicts.

**⚠️ Alpha software. Use at your own risk. Always verify results with a certified ski technician before drilling anything.**

---

## What it does

1. **Calibrate** — you click 2–4 marks on a measuring tape laid alongside the ski. With 3+ marks the app corrects for camera parallax (perspective distortion along the ski axis) before doing any measurements. You also mark the ski's centre mount reference and optionally draw the ski axis for higher accuracy.
2. **Mark holes** — click on the rectified image to mark every existing drill hole. No auto-detection; manual placement is deliberate (see [Removed features](#removed-for-now)).
3. **Analyze** — the app projects every binding template onto the ski coordinate system and classifies each hole as reusable, new, or conflicting. Results are sorted by compatibility.
4. **Explore** — click any binding row to overlay its template on the photo. Sliders let you adjust BSL, heel position, and mounting point offset in real time.

---

## Screenshots

![Calibration](https://skipass.fr/p/imagebank/2/4/1/241806-bxKTu3zX.pn "Calibration") ![Holes marking](https://skipass.fr/p/imagebank/2/4/1/241807-4183A1Rd.png "Holes marking") ![Binding analysis](https://skipass.fr/p/imagebank/2/4/1/241808-pz0Hb4w8.png "Binding analysis")
---

## Features

- 📏 **Parallax correction** — 3 or 4 tape marks fit a 1-D projective transform along the ski axis and remap the image before any measurements. Significantly improves accuracy for photos taken at an angle.
- 🗺️ **Axis calibration** — optional edge-to-edge transversal lines constrain the ski axis more precisely than the tape alone; the mounting point is projected onto the computed centreline.
- 🖊️ **Manual hole editor** — click to add, click again to remove. Simple and reliable.
- 🎿 **13 binding templates** — alpine and touring/LT; see full list below.
- 🔍 **Interactive overlay** — BSL, heel offset, and mount offset sliders update the overlay in real time; a second cross shows the offset mounting point visually without moving the calibration reference.
- 📐 **Debug grid** — toggle a 1 cm or ½ cm grid aligned to the ski axis to verify calibration visually.
- 🐳 **Docker deployment** — one `docker compose up --build` and you're done.

---

## Quick Start (Docker)

```bash
git clone https://github.com/your-username/ski-binding-conflict.git
cd ski-binding-conflict

docker compose up --build -d

# Open http://localhost:8000
```

No API keys or environment variables required.

---

## Local Development

```bash
# Python 3.11+ required
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8765 --reload

# Open http://localhost:8765
```

---

## How to Use It

### Step 1 — Upload & Calibrate

Take a photo of your ski laid flat with a measuring tape running alongside it. The tape should have clearly visible tick marks. Phone photos work fine; uniform lighting helps.

Click the calibration tools in order:

| Button | What to do |
|--------|------------|
| **① Tape** | Click 2–4 marks at equal known spacing (e.g. every 100 mm, heel → tip). 3+ marks enable parallax correction. |
| **② Mount pt** | Click the ski's centre mark — the half-sole reference line printed on the ski. |
| **③ Axis L1** *(optional)* | Click one ski edge, then the opposite edge at the heel zone. Defines the ski axis better than the tape alone. |
| **④ Axis L2** *(optional)* | Same as L1 but at the tip zone. Together L1+L2 give a precise centreline. |
| **↩ Flip axis** | If the orange **→TIP** arrow points the wrong way, click this. |

Click **Calibrate →**. The app sends the image to the server, applies the perspective correction if you gave 3+ tape marks, and shows you the rectified image.

### Step 2 — Mark Holes

Click on the rectified image to mark each existing drill hole. Click an existing circle to remove it. There is no auto-detection.

Set your **Boot sole length (BSL)**, binding **category**, and **minimum hole separation** (default 14 mm), then click **Analyze →**.

### Step 3 — Results

The results table lists every binding sorted by compatibility:

- **OK** — all template holes either reuse existing holes or land in undamaged areas
- **CONFLICT** — one or more template holes fall too close to an existing hole

Click any row to overlay the binding template on your ski photo. The sliders at the top let you adjust:

- **BSL** — boot sole length (changes toe/heel separation)
- **Heel offset** — if the binding's heel unit has a longitudinal adjustment range
- **Mount offset** — shift the entire binding forward/back relative to your calibration point

The **yellow cross** always marks your original calibration point. When mount offset ≠ 0 a **white cross** appears at the effective mounting position.

The **debug grid** (1 cm or ½ cm) overlays a coordinate grid aligned to the ski axis, centred on the mount point — useful for verifying that calibration and parallax correction look right.

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

Templates live in `templates/binding_db.json`. The format is self-documented in the file. Since the templates directory is mounted as a Docker volume you can add or edit bindings without rebuilding the container.

---

## Removed for Now

These features existed in earlier versions and were deliberately removed. They may return.

| Feature | Status | Reason |
|---------|--------|--------|
| **Auto hole detection** (OpenCV HoughCircles / blob) | Removed | Results were unreliable — too many false positives on dark skis or complex graphics. Manual marking is slower but actually correct. |
| **AI hole detection** (Ollama / Claude Vision) | Removed | Code existed but was largely untested in practice. Removed rather than ship something misleading. |
| **Multi-BSL range probing** (exact BSL ±30 mm) | Removed | Added noise to the results table without a clear UX benefit given the new manual workflow. |
| **Access token auth** | Removed | Simplified the deployment. If you expose this publicly, put a reverse proxy in front of it. |

---

## Limitations

- **Only 13 bindings are active.** Source PDFs exist for ~33, but the remaining ~20 have not been transcribed yet. The database only includes bindings whose hole coordinates have been worked out and confirmed — see the [Binding Database](#binding-database) section for how to contribute.
- **Parallax correction is 1-D only.** It corrects perspective distortion along the ski axis, based on the assumption that the camera is roughly centred over the ski laterally. Extreme lateral camera offsets are not handled.
- **No warranty whatsoever.** Drilling ski holes in the wrong place can damage the ski, compromise binding retention, and create a safety hazard. This tool is a planning aid, not a substitute for a qualified ski technician with a calibrated drill jig.

---

## Project Status

**Alpha. Personal project.** Built because I needed it; shared because someone else might too.

Development continues when time allows. No roadmap, no support SLA. Bug reports and binding template PRs are welcome.

---

## Contributing

- **New binding templates** — most useful contribution. Include the source PDF, note whether you verified it on a real ski, and open a PR.
- **Bug reports** — include a photo (or describe it), what you expected, and what happened.
- **Code** — keep it simple.

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

*Built with FastAPI, OpenCV, and manufacturer PDF drill templates.*
