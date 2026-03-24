# PCAT Workstation

Pericoronary Adipose Tissue (PCAT) analysis workstation for cardiac CT (CCTA). Automatically segments coronary arteries, extracts vessel wall contours, and computes FAI (Fat Attenuation Index) statistics per vessel.

> **FAI risk threshold: -70.1 HU.** Values above this (less negative) indicate higher cardiovascular inflammation risk (Oikonomou et al., Lancet 2018; CRISP-CT trial).

---

## Installation

```bash
git clone https://github.com/MolloiLab/PCAT.git
cd PCAT
pip install -r requirements.txt
```

**TotalSegmentator licence (required for auto seed detection):**
Get a free academic licence at https://backend.totalsegmentator.com/license-academic/ and set:
```bash
export TOTALSEG_LICENSE=<your-key>
```

**Apple Silicon users:** PyTorch with MPS acceleration is auto-detected. Install with `pip install torch` for faster processing.

---

## Quick Start

```bash
cd PCAT
python -m pcat_workstation.main
```

1. Click **Import DICOM** (left panel) and select a DICOM folder
2. Click **Run** (right panel) to execute the next pipeline stage, or **Run All** (toolbar) for the full pipeline
3. Seeds are auto-detected and editing is enabled immediately
4. Review results in the analysis dashboard and export as PDF or DICOM

---

## User Interface

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [LAD] [LCx] [RCA]  hints...  W/L:[preset]  ▶▶Run All  Export│  ← Toolbar
├──────────┬──────────────────────────────────┬───────────────┤
│          │  Axial        │  Coronal         │               │
│  DICOM   │───────────────┼──────────────────│  Pipeline     │
│  Browser │  Sagittal     │  CPR Viewer      │  Progress     │
│          │               │  ┌──┐┌──┐┌──┐   │  Vessel       │
│          │               │  │A ││B ││C │   │  Results      │
│          │               │  └──┘└──┘└──┘   │               │
├──────────┴──────────────────────────────────┴───────────────┤
│  ▲ Analysis  [HU Histogram]  [Radial Profile]               │  ← Dashboard
└─────────────────────────────────────────────────────────────┘
```

### Toolbar

| Control | Description |
|---------|-------------|
| **LAD / LCx / RCA** | Select vessel — navigates all views to its ostium (keyboard: 1/2/3) |
| **Hint bar** | Shows editing shortcuts at a glance |
| **Zoom** | Pinch-to-zoom on trackpad or Ctrl+scroll (synced across all views) |
| **W/L preset** | CT Vascular, Soft Tissue, Mediastinum, Lung, Bone |
| **Run All** | Execute entire pipeline from scratch (Ctrl+Shift+R) |
| **Export** | Generate PDF report |

### MPR Views (2x2 Grid)

The top-left, top-right, and bottom-left panels show **axial**, **coronal**, and **sagittal** slices following ImageJ/radiology convention:
- **Scroll** to change slice
- **Left-click** to navigate crosshair (synced across all views)
- **Right-drag** to adjust window/level
- **Pinch-to-zoom** on trackpad, or **Ctrl+scroll** (synced across views)

### CPR Viewer (Bottom-Right)

The Curved Planar Reformation viewer shows the straightened vessel with interactive controls:

#### CPR Toolbar
| Control | Description |
|---------|-------------|
| **Angle slider** (0-360) | Rotate the cutting plane around the vessel centerline |
| **Straightened / Stretched** | Toggle display mode (Straightened fills panel; Stretched preserves 1:1 mm aspect ratio for accurate measurement) |
| **Tool selector** | Navigate, W/L, Pan, Zoom, or Measure |

#### CPR Interactions
| Action | Effect |
|--------|--------|
| **Left-click** (Navigate tool) | Move needle to that position along the vessel |
| **Drag yellow A/C line** | Adjust spacing between the 3 cross-section positions |
| **Drag cyan B line** | Move main needle position |
| **Ctrl+scroll** | Change A-B-C interval |
| **Shift+scroll** | Rotate cutting plane by 5 degrees |
| **Scroll** | Step needle along vessel |
| **Right-drag** | Adjust window/level (always active) |

#### Cross-Section Views (A, B, C)

Three perpendicular cross-sections are displayed on the right side of the CPR panel:
- **B** (cyan) — main needle position
- **A** (yellow) — proximal to B
- **C** (yellow) — distal to B

Each shows: vessel lumen contour (white), VOI ring (yellow dashed), arc-length position, equivalent radius, and distance from B.

#### Measurement Tool

Select **Measure** from the tool dropdown:
1. Click and drag on the CPR image to draw a measurement line
2. Distance is displayed in mm (computed from the CPR's physical scale)
3. Press **Delete** to remove the last measurement
4. Measurements clear when switching vessels or rotating

> In **Stretched** mode, distances are accurate in any direction. In **Straightened** mode, only horizontal (lateral) and vertical (arc-length) distances are accurate.

---

## Seed Editing

Seeds are auto-detected when the pipeline runs and **edit mode activates automatically**. No toggle needed.

| Action | Effect |
|--------|--------|
| **Left-click near seed** | Select it (yellow highlight) |
| **Drag selected seed** | Move to new position |
| **Enter** | Add waypoint at current crosshair position |
| **Backspace / Delete** | Delete selected waypoint |
| **Ctrl+Z** | Undo |
| **Ctrl+Shift+Z / Ctrl+Y** | Redo |
| **Left/Right arrows** | Cycle through vessels |

Seed markers: **cube** = ostium (vessel origin), **sphere** = waypoints. Centerlines are drawn through all seeds in vessel color.

---

## Pipeline Stages

The pipeline runs in 6 stages. Use **Run** to execute one stage at a time, or **Run All** for the full sequence.

| Stage | Description | Time |
|-------|-------------|------|
| **Import** | Load DICOM volume | < 1s |
| **Seeds** | Auto-detect coronary ostia + waypoints via TotalSegmentator | 30-60s |
| **Centerlines** | Extract + clip vessel centerlines, estimate radii | 2-5s |
| **Contours** | Polar-transform vessel wall detection + CPR generation | 5-10s |
| **PCAT VOI** | Build pericoronary VOI mask (CRISP-CT: 1mm gap + 3mm ring) | 3-5s |
| **Statistics** | Compute FAI, fat fraction, HU histogram, radial profile | < 1s |

### Seed Detection Algorithm

TotalSegmentator segments the coronary arteries, then:
1. **Aorta detection** — runs aorta segmentation for accurate ostium placement (cached after first run)
2. **Vessel assignment** — assigns LAD/LCx/RCA by angular position relative to the aorta center (not by component size)
3. **Ostium placement** — places ostium at the skeleton endpoint nearest the aorta surface
4. **Waypoints** — 3 evenly spaced points along the proximal 45mm of each vessel

### VOI Geometry

Two modes (configurable in Settings):
- **CRISP-CT** (default): 1mm gap from vessel wall + 3mm ring width (Oikonomou et al., Lancet 2018)
- **Scaled**: VOI extends to `pcat_scale x r_eq` from the vessel wall (default 3x)

---

## Analysis Dashboard

After the pipeline completes, the collapsible dashboard at the bottom shows:
- **HU Histogram** — distribution of HU values within the PCAT VOI, with -70.1 HU risk threshold line
- **Radial Profile** — mean HU vs. distance from vessel wall

---

## Export

### PDF Report (File > Export...)
Multi-page PDF with:
- Summary page: patient info + per-vessel FAI table with risk classification
- Per-vessel pages: CPR image, HU histogram, statistics box

### DICOM Export (File > Export DICOM...)
Exports CPR images as DICOM Secondary Capture objects:
- 16-bit grayscale with patient metadata
- All vessels grouped under one Study Instance UID for PACS compatibility

### Save/Load Path (File > Save Path... / Load Path...)
Export the current centerline path + seeds as a JSON file. Reload later to restore the exact vessel path without re-running the pipeline.

---

## Settings

**File > Settings** opens the configuration dialog:

| Setting | Default | Description |
|---------|---------|-------------|
| VOI Mode | CRISP-CT | Fixed ring (CRISP-CT) or scaled (pcat_scale x r_eq) |
| CRISP Gap | 1.0 mm | Distance from vessel wall to VOI inner boundary |
| CRISP Ring | 3.0 mm | Width of the VOI ring |
| Scaled Factor | 3.0x | VOI outer boundary as multiple of r_eq |
| LAD/LCx Length | 40 mm | Proximal segment length for analysis |
| RCA Length | 40 mm | Proximal segment length |
| RCA Start | 10 mm | Skip first 10mm (aortic root) |
| FAI Min/Max | -190 / -30 HU | Fat attenuation window |
| Default W/L | 800 / 200 | CT vascular window preset |

Settings are saved to `~/.pcat_workstation/config.json`.

---

## Batch Processing

**Pipeline > Batch Processing** opens a dock panel for processing multiple patients:

1. Click **Add Folders** to queue DICOM directories
2. Set the output directory
3. Click **Start Batch** — patients are processed sequentially
4. Each patient shows status: pending, running, completed, or failed

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **1 / 2 / 3** | Select LAD / LCx / RCA |
| **Ctrl+R** | Run next pipeline stage |
| **Ctrl+Shift+R** | Run all stages |
| **Ctrl+O** | Import DICOM |
| **Ctrl+E** | Export PDF |
| **Ctrl+Q** | Quit |
| **Ctrl+Z** | Undo (seed editing) |
| **Ctrl+Shift+Z** | Redo (seed editing) |
| **Enter** | Add waypoint |
| **Backspace** | Delete selected waypoint |
| **Delete** | Remove last measurement line |
| **Scroll** | Change slice / move CPR needle |
| **Ctrl+Scroll** | Zoom (MPR) / change A-B-C interval (CPR) |
| **Shift+Scroll** | Rotate CPR cutting plane |
| **Pinch-to-zoom** | Zoom in/out on trackpad (synced across MPR views) |

---

## CLI Pipeline (Advanced)

The standalone CLI pipeline is still available for scripting and batch processing:

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds
```

For headless/batch mode:
```bash
python pipeline/run_pipeline.py --batch --auto-seeds --skip-editor --skip-cpr-browser
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Font warning on launch | Fixed — uses Helvetica instead of sans-serif |
| Left/right reversed in axial view | Fixed — camera looks from inferior (feet) per radiology convention |
| TotalSegmentator fails | Place seeds manually, or set `TOTALSEG_LICENSE` env var |
| No vessels detected | Check DICOM is a contrast-enhanced cardiac CT (CCTA) |
| NaN mean HU | Vessel may be heavily calcified; no fat voxels in VOI |
| CPR looks wrong vs Horos | Verify CPR orientation: arc-length runs vertically (top=proximal) |
| Slow on CPU | Install PyTorch (`pip install torch`) for MPS acceleration on Apple Silicon |

---

## Building for Distribution

```bash
# Build macOS .app bundle + DMG installer
./scripts/build_dmg.sh
```

Requires `pip install pyinstaller`. Output: `dist/PCAT_Workstation_1.0.0.dmg`

---

## References

- Oikonomou EK et al. Non-invasive detection of coronary inflammation using computed tomography and prediction of residual cardiovascular risk. *Eur Heart J* 2018.
- Oikonomou EK et al. A novel machine learning-derived radiotranscriptomic signature of perivascular fat improves cardiac risk prediction using coronary CT angiography. *Lancet* 2018 (CRISP-CT).
- Kanitsar A et al. CPR — Curved Planar Reformation. *IEEE Visualization* 2002.

---

*Molloi Lab — UC Irvine*
