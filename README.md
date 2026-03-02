# ⛷️ Ski Binding Conflict Detector

> *"Will my new bindings fit without drilling new holes?" — finally, a tool that attempts to answer this.*

A web application that analyses a photo of a ski, detects existing binding drill holes, and tells you which bindings can be mounted without conflicts — and which ones would require new holes, partial reuse, or are just incompatible.

**⚠️ Alpha software. Use at your own risk. Always verify results with a certified ski technician before drilling anything.**

---

## What does it do?

You upload a photo of your ski (with a measuring tape next to it), mark a few calibration points, and the app:

1. Detects existing drill holes using OpenCV computer vision
2. Lets you manually add/correct holes
3. Checks every binding in the database against your existing holes
4. Tells you which bindings are **mountable as-is**, which need **new holes only**, and which create **conflicts** (too close to existing holes)
5. Optionally tests at ±30mm BSL range to catch cases where a slightly different boot sole length position would fit better — useful when a ski has been drilled for multiple boots over the years

The overlay image shows the binding template on your actual ski photo so you can see exactly where the holes would land.

---

## Screenshots

*(Coming eventually. For now: upload photo → click some points → get a table of results and an annotated overlay. You'll figure it out.)*

---

## Features

- 🔍 **OpenCV hole detection** — HoughCircles + blob detection, works offline, no API needed
- ✏️ **Manual hole editor** — add or remove holes by clicking on the photo
- 📏 **Manual calibration** — set scale from a measuring tape, mark the mounting point, set ski axis direction
- 🔄 **Multi-BSL analysis** — tests each binding at your BSL ±30mm in 10mm steps
- 🎿 **33 binding templates** — alpine and touring/LT, see full list below
- 🤖 **AI fallback (optional)** — Ollama (local, free) or Claude Vision API for automatic hole detection and scale reading from the photo
- 🐳 **Docker deployment** — one `docker compose up --build` and you're done
- 🔑 **Optional access token** — for when you put it on a server and don't want strangers poking at it

---

## Quick Start (Docker)

```bash
# Clone the repo
git clone https://github.com/derdide/solemate.git
cd solemate

# Create your config
cp .env.example .env
# Edit .env — at minimum set APP_TOKEN to something random:
# python -c "import secrets; print(secrets.token_urlsafe(32))"

# Build and run
docker compose up --build -d

# Open http://localhost:8000
```

That's it. OpenCV works out of the box with no API keys.

---

## Local Development (no Docker)

```bash
# Python 3.11+ required
pip install -r requirements.txt

# Run with hot-reload
uvicorn main:app --host 0.0.0.0 --port 8765 --reload

# Open http://localhost:8765
```

---

## Configuration

Copy `.env.example` to `.env` and edit:

```env
# Access token for the web UI (leave empty to disable auth in dev)
APP_TOKEN=your-long-random-string-here

# Optional: Ollama Vision (local AI, no API key needed)
# Point to your Ollama host — tried first when set
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-vl:7b        # recommended; llava also works

# Optional: Claude Vision API (cloud, needs key)
# Only called if Ollama is not set or returned nothing
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-opus-4-6

# Set to "false" to skip Claude entirely and use OpenCV + manual calibration only
USE_CLAUDE_FALLBACK=true
```

**Without any API keys** the app works fine — you just do the calibration manually (takes about 30 seconds once you know what to click).

### Ollama tip

`qwen2.5-vl:7b` is the best local model for this task — it follows structured JSON instructions reliably and reads measuring tape numbers well.

```bash
ollama pull qwen2.5-vl:7b
```

---

## How to Use It

### Step 1 — Upload a photo

Take a photo of your ski laid flat with a measuring tape alongside it. The tape should be clearly visible with legible tick marks. Uniform lighting helps. Phone photos work fine.

### Step 2 — Set calibration

Click the calibration buttons in order:

1. **① Tape** — click two clearly labelled tape marks (e.g. 10cm and 20cm apart), enter the distance in mm
2. **② Mounting point** — click the centre mark on the ski (the half-sole line)
3. **③ Axis L1 / ④ Axis L2** *(optional but recommended)* — click across the ski at two locations to define the ski axis precisely. The orange **→HEEL** arrow shows the computed direction; use **↩ Flip axis** if it points the wrong way.

### Step 3 — Analyze

Enter your boot sole length (BSL in mm), choose a binding category, hit **Analyze**.

The results table shows every binding sorted by compatibility. Click a row to see the binding template overlaid on your ski photo.

---

## Binding Database

33 bindings covering alpine and touring/LT categories. Templates are sourced from manufacturer PDF drill guides.

**Verification status**: `✓` = template cross-checked against reference data. `⚠` = template entered but not independently verified — treat with extra caution.

*Honest disclaimer: "verified" largely means I didn't make a typo transcribing the PDF. It does not mean I physically drilled a ski and measured the result. Some of these I have definitely not verified personally. Use your own judgement.*

### Alpine

| Status | Binding |
|--------|---------|
| ✓ | Marker Jester / Griffon / Squire |
| ✓ | Marker Duke / Baron / Tour F10/F12 |
| ✓ | Marker Duke EPF |
| ✓ | Marker M-Series |
| ✓ | Marker Ten Free |
| ✓ | Salomon Alpine Bindings |
| ✓ | Salomon STH2 / Warden MNC |
| ⚠ | Salomon Warden 11 |
| ✓ | Salomon Guardian / Atomic Tracker |
| ✓ | Salomon Rental (SC) |
| ✓ | Tyrolia / Head Alpine |
| ⚠ | Tyrolia / Head AAAttack13 Demo/Rental |
| ✓ | Rossignol Axial1 / Look Pivot (pre-2005) |
| ✓ | Rossignol Axial2 / Look PX |
| ✓ | Rossignol FKS / Look Pivot (post-2010) |
| ✓ | Rossignol Axial Racing / Look PX Racing |
| ✓ | Naxo NX01/NX21 (Alpine) |

### Touring / LT

| Status | Binding |
|--------|---------|
| ✓ | Marker Kingpin |
| ✓ | Tyrolia / Head AAAmbition |
| ✓ | Tyrolia / Head AAAmbition Carbon |
| ✓ | Dynafit Radical (TLT) V1 |
| ✓ | Dynafit Radical 2 |
| ✓ | Dynafit TLT Superlite |
| ✓ | Dynafit TLT Vertical/Speed / G3 Onyx |
| ✓ | Dynafit Beast 14 |
| ✓ | Dynafit Beast 16 |
| ✓ | G3 Ion / Zed |
| ✓ | Fritschi Diamir Freeride / Freeride Plus |
| ✓ | Fritschi Diamir FreeridePro + Eagle (2010) |
| ✓ | Fritschi Diamir Vipec |
| ✓ | Naxo NX01/NX21 (Touring) |
| ⚠ | Salomon MTN |
| ⚠ | Salomon Shift MNC 13 |

### Adding bindings

Templates live in `templates/binding_db.json`. The format is documented in the file header. Since the templates directory is mounted as a Docker volume, you can add or edit bindings without rebuilding the container. PRs with new verified templates are welcome.

---

## Limitations & Known Issues

This is **alpha software** built for a very specific personal need. It has been tested on a small number of skis in limited conditions. Here's what you should know:

- **Hole detection is imperfect.** OpenCV struggles with dark skis, busy graphics, poor lighting, or small photos. The manual hole editor exists precisely because auto-detection will sometimes miss holes or find phantom ones. Always review the detected holes before relying on results.

- **AI fallbacks (Ollama / Claude) are largely untested.** The code is there and it compiles, but real-world testing has been minimal. Ollama results in particular depend heavily on which model you use and how it was trained.

- **The binding template database is incomplete.** 33 bindings is a start, not a finish. Notably missing: most Atomic/Salomon touring bindings, older Fritschi models, Plum, Hagan, and many others. Missing your binding? Add it from the PDF.

- **BSL range testing (±30mm) helps but is not magic.** It catches cases where the ski was previously mounted for a different boot size. It does not account for the heel piece's longitudinal adjustment range in a meaningful physical way.

- **No warranty whatsoever.** Drilling ski holes in the wrong place can damage the ski, compromise binding retention, and create a safety hazard. This tool is a *planning aid*, not a replacement for a qualified ski technician with a drill jig.

---

## Project Status

**Alpha. Personal project.** I built this because I needed it, and I'm sharing it because someone else might find it useful too.

I may continue developing it. I may not. No promises. If you file an issue I might look at it. If you submit a clean PR with a verified binding template I'll probably merge it. Beyond that: no guarantees, no support SLA, no roadmap.

If you do something cool with it, I'd love to hear about it — but you're under no obligation.

---

## Contributing

- **New binding templates** — most useful contribution. Include the source PDF if possible, and note whether you've physically verified the template on a real ski.
- **Bug reports** — include the ski photo (or a description of it), what you expected, and what happened.
- **Code improvements** — keep it simple. This is not a startup.

---

## License

MIT License — do whatever you want with it, just keep the copyright notice.

```
MIT License

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

*Built with FastAPI, OpenCV, and a pile of PDF drill templates. Tested on an embarrassingly small number of actual skis.*
