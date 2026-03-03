# PCAT — Pericoronary Adipose Tissue Pipeline

Automatically measures fat around coronary arteries from a cardiac CT scan (CCTA). Outputs CPR images, HU histograms, radial profiles, and a per-vessel FAI summary.

---

## What does it do?

1. Finds your coronary arteries (LAD, LCX, RCA) automatically via TotalSegmentator
2. Lets you review and refine the coronary seed locations
3. Computes Frangi vesselness filtering and extracts centerlines
4. Extracts real vessel wall contours via polar-transform boundary detection
5. Lets you correct contours in an interactive clinical editor
6. Builds PCAT VOI from contours (3× vessel radius) and computes FAI
7. Generates CPR images, histograms, and statistics

> **FAI risk threshold: −70.1 HU.** Values above this (less negative) = higher cardiovascular inflammation risk (Oikonomou et al., Lancet 2018).

---

## Setup (one time)

```bash
git clone https://github.com/MolloiLab/PCAT.git
cd PCAT
pip install -r requirements.txt
```

**Optional (recommended on Apple Silicon):** Install PyTorch for GPU-accelerated Frangi filtering via Metal Performance Shaders. Without it, the pipeline falls back to CPU (scikit-image).

```bash
pip install torch
```

Get a free TotalSegmentator research licence at:
`https://backend.totalsegmentator.com/license-academic/`

---

## Standard Workflow (semi-automatic, one command)

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds
```

The pipeline runs all stages automatically and pauses at each interactive review step:

### Stage 1 — Auto seed detection (~30–60 s)

TotalSegmentator segments the cardiac CT and detects the coronary artery ostia and distal endpoints. Seeds are saved to `seeds/`.

### Stage 2 — Seed Reviewer

A window opens showing 3 MPR planes (axial, coronal, sagittal) with the auto-detected seeds overlaid. If reviewed seeds already exist from a previous run, this step is skipped automatically.

| Key / Action | Effect |
|---|---|
| `1` / `2` / `3` | Switch active vessel (LAD / LCX / RCA) |
| Click near a seed | Drag to reposition |
| Click elsewhere | Add new waypoint |
| `d` | Delete the nearest waypoint |
| `u` | Undo last action |
| `r` | Reset current vessel to original seeds |
| `c` | Clear warning messages |
| `w` / `W` | Window width wider / narrower |
| `l` / `L` | Window level brighter / darker |
| `s` | **Save seeds (JSON) & continue pipeline** |
| `q` | Quit without saving |
| Scroll wheel | Change slice |

### Stage 3 — Frangi vesselness filtering (~120 s)

Computes a multi-scale Frangi vesselness filter on a ROI-cropped volume around the seed points. Uses MPS (Metal Performance Shaders) on Apple Silicon if PyTorch is installed, otherwise falls back to CPU via scikit-image.

### Stage 4 — Per-vessel processing (~5 s total)

For each vessel (LAD, LCX, RCA) the pipeline automatically:

1. Extracts the centerline from the vesselness map via FMM/Dijkstra shortest path (with cubic-spline fallback if Frangi yields too few points)
2. Clips to the proximal segment (LAD/LCX: 5–45 mm, RCA: 10–50 mm)
3. Estimates vessel radii along the centerline
4. Extracts vessel wall contours via polar-transform boundary detection using half-maximum descent with Chan-Vese level-set fallback

### Stage 5 — Centerline verification

Generates a static overlay image showing extracted centerlines on top of the CT volume, with TotalSegmentator coronary segmentation mask if available. Saved to `plots/`.

### Stage 6 — Contour Editor

Opens a clinical GUI for reviewing and correcting the auto-extracted vessel wall contours before PCAT computation.

**Vessel colors:** LAD = red-orange · LCX = blue · RCA = green

| Key / Action | Effect |
|---|---|
| `M` | **Toggle edit mode** (Scissors ↔ Refine) |
| `A` | **Auto-snap boundary** (re-detect via gradient analysis) |
| `1` / `2` / `3` | Switch active vessel |
| `←` / `→` | Navigate ±1 cross-section position |
| `↑` / `↓` | Navigate ±5 positions |
| Left-drag (Scissors) | Draw freehand lasso to fill/erase region |
| Left-drag (Refine) | Smooth contour deformation with Gaussian falloff |
| Right-drag | Scissors/lasso tool (backward compat) |
| `E` | Toggle scissors mode (fill ↔ erase) |
| `I` | Fill between slices (interpolate modified positions) |
| `R` | Reset current contour to auto-detected |
| `Space` | Toggle contour visibility |
| `V` | Toggle VOI ring visibility |
| Scroll wheel | Zoom cross-section view |
| `S` | **Save corrected contours & continue pipeline** |
| `Q` | Quit without saving (uses auto-detected contours) |
- **Left panel:** CPR cross-section with vessel wall contour
- **Right panel:** Vessel overview — r_eq profile along arc-length with modification markers
- **Separate window:** 3D pyvista visualization with colored vessel meshes and semi-transparent fat volume
- **Two editing modes:** Scissors (default) for bulk fill/erase with freehand lasso; Refine for smooth point-by-point contour adjustment
- **Auto-snap:** Press `A` to re-detect the vessel boundary using gradient analysis from the CT image — eliminates star-shaped artifacts

### Stage 7 — VOI construction & FAI computation (~3 s)

For each vessel, builds the PCAT VOI by morphological dilation of the corrected vessel wall contours:
- Computes equivalent radius: `r_eq = sqrt(area / pi)`
- Dilates by `d = 3 × r_eq` to create the pericoronary region
- Applies fat threshold: −190 to −30 HU
- Computes FAI statistics (mean HU, fat fraction, voxel count)

### Stage 8 — CPR Browser (opens once per vessel)

After VOI computation, an interactive CPR browser opens for each vessel sequentially.

| Key / Action | Effect |
|---|---|
| Arc-length slider | Move the cross-section needle along the vessel |
| Rotation slider | Rotate the cutting plane (0–360°) |
| Click on CPR image | Jump needle to that vessel position |
| `←` / `→` or `↑` / `↓` | Step needle by one point |
| Scroll wheel | Rotate cutting plane by ±5° |
| `a` | Toggle anchor mode (click to place anchors on CPR) |
| `p` | Apply anchors and print anchor data |
| `r` | Reset rotation to 0° |
| `q` | Close and continue to next vessel |

### Stage 9 — Export & summary

1. Per-vessel `.raw` VOI volumes + metadata JSON
2. Combined all-vessel VOI exported as `.raw`
3. CPR images: FAI overlay, vessel wall overlay, DICOM secondary capture
4. HU histograms and radial HU profiles per vessel
5. Summary bar chart and `*_results.json`

---

## Estimated total runtime

| Stage | Time |
|-------|------|
| Auto seeds (TotalSegmentator) | ~30–60 s |
| Seed review | Interactive |
| Frangi vesselness (MPS GPU) | ~120 s |
| Per-vessel processing | ~5 s |
| Contour editor | Interactive |
| VOI + outputs | ~3 s |
| CPR browsers | Interactive |
| **Total (non-interactive)** | **~3 min** |

---

## Fully Automatic Mode (batch / headless)

Use this for bulk processing or servers with no display.

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds \
    --skip-editor \
    --skip-cpr-browser
```

**Run all patients at once:**

```bash
python pipeline/run_pipeline.py --batch --auto-seeds --skip-editor --skip-cpr-browser
```

**Legacy VOI mode** (uses estimated circles instead of polar-transform contours):

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200.json \
    --output output/patient_1200 \
    --prefix patient1200 \
    --legacy-voi
```

> Do not use fully automatic mode for clinical reporting without a separate manual review step.

---

## Output files

All outputs are organized into subdirectories under the `--output` path:

```
output/patient_1200/
├── patient1200_results.json          # Per-vessel FAI statistics
├── cpr/
│   ├── patient1200_LAD_cpr_fai.png   # CPR with FAI color overlay
│   ├── patient1200_LAD_cpr_wall.png  # CPR with vessel wall + VOI boundary lines
│   └── patient1200_LAD_cpr_hu.dcm    # CPR as DICOM secondary capture
├── plots/
│   ├── patient1200_centerline_verification.png  # Centerline + TotalSeg overlay
│   ├── patient1200_LAD_hu_histogram.png         # HU distribution in VOI
│   ├── patient1200_LAD_radial_profile.png       # Fat HU vs. distance from wall
│   └── patient1200_summary.png                  # Bar charts: FAI, fat fraction, voxel count
└── raw/
    ├── patient1200_contour_data.npz             # Vessel wall contours (for contour editor)
    ├── patient1200_LAD_voi.raw                  # Per-vessel VOI volume
    ├── patient1200_LAD_voi_metadata.json        # Volume dimensions, spacing, origin
    ├── patient1200_combined_voi.raw             # All-vessel combined VOI
    └── patient1200_combined_voi_metadata.json
```

Each vessel (LAD, LCX, RCA) produces its own set of `cpr/`, `plots/`, and `raw/` files.

---

## Reading CPR images

- **Vessel runs top → bottom.** Top = ostium (origin), bottom = distal end.
- **Yellow/red regions** (in `*_cpr_fai.png`) = pericoronary fat (FAI range −190 to −30 HU). Yellow = more negative HU (healthier). Red = less negative (more inflamed).
- **Cyan contour** (in `*_cpr_wall.png`) = detected vessel lumen boundary.
- **Green dashed lines** (in `*_cpr_wall.png`) = VOI outer boundary (3× lumen radius).

---

## Standalone tools

These can be run independently outside the pipeline:

| Tool | Usage |
|------|-------|
| `pipeline/seed_picker.py` | Manual seed placement: `python pipeline/seed_picker.py --dicom <dir> --output seeds/patient.json` |
| `pipeline/seed_reviewer.py` | Review/edit seeds: `python pipeline/seed_reviewer.py --dicom <dir> --seeds <json> --output <json>` |
| `pipeline/cpr_browser.py` | Browse CPR interactively: `python pipeline/cpr_browser.py --dicom <dir> --seeds <json> --vessel LAD` |
| `pipeline/contour_editor.py` | Edit vessel contours: `python pipeline/contour_editor.py --dicom <dir> --contour-data <npz> --output <dir> --prefix <name>` |
| `pipeline/coronary_contour_editor.py` | Legacy contour editor: `python pipeline/coronary_contour_editor.py --dicom <dir> --data <npz> --output <dir> --prefix <name>` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "too few centerline points" | Re-run and adjust seeds in the Seed Reviewer. The pipeline uses cubic-spline interpolation as fallback. |
| NaN mean HU in results | No fat voxels in VOI; vessel may be heavily calcified |
| "only N vessels found" warning | Add the missing vessel manually in the Seed Reviewer |
| Seed Reviewer / Contour Editor won't open | Add `--skip-editor --skip-cpr-browser` (headless server) |
| TotalSegmentator fails | Place seeds manually with `seed_picker.py` (see Standalone Tools) |
| Frangi is slow (~120 s on CPU) | Install PyTorch (`pip install torch`) for MPS GPU acceleration on Apple Silicon |
| `s` key opens PNG save dialog | Update to latest version — this was a matplotlib keybinding conflict (fixed) |

---

## Patient data included

| Patient | DICOM folder | Slices |
|---------|-------------|--------|
| 1200 | `Rahaf_Patients/1200.2/` | 405 |
| 2 | `Rahaf_Patients/2.1/` | 149 |
| 317 | `Rahaf_Patients/317.6/` | 399 |
| 161 | `Rahaf_Patients/161.6/` | — |

To add a new patient, change `--dicom` to the new DICOM folder and `--output` to a new folder.

---

## Adding multiple patients (batch)

Edit `PATIENT_CONFIGS` in `pipeline/run_pipeline.py`:

```python
PATIENT_CONFIGS = [
    {"patient_id": "1200", "dicom": "Rahaf_Patients/1200.2", "output": "output/patient_1200", "prefix": "patient1200"},
    {"patient_id": "999",  "dicom": "New_Patients/patient999", "output": "output/patient_999",  "prefix": "patient999"},
]
```

Then run:

```bash
python pipeline/run_pipeline.py --batch --auto-seeds
```
