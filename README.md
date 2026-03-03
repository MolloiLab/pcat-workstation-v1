# PCAT — Pericoronary Adipose Tissue Pipeline

Automatically measures fat around coronary arteries from a cardiac CT scan (CCTA). Outputs CPR images, HU histograms, radial profiles, and a per-vessel FAI summary.

---

## What does it do?

1. Finds your coronary arteries (LAD, LCX, RCA) automatically
2. Lets you review and refine the coronary seed locations
3. Extracts centerlines, estimates vessel radii, and builds pericoronary VOIs
4. Lets you review and adjust centerlines, vessel wall contours, and PCAT volumes
5. Lets you browse interactive CPR images per vessel
6. Generates the PCAT (pericoronary fat) volume and measures the **Fat Attenuation Index (FAI)**
7. Saves images and statistics you can review and report

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
| Click on any plane | Move the selected seed to that location |
| `d` | Delete the nearest waypoint |
| `u` | Undo last action |
| `r` | Reset current vessel to original seeds |
| `c` | Clear warning messages |
| `w` / `W` | Increase / decrease window width |
| `l` / `L` | Increase / decrease window level |
| `s` | **Save seeds & continue pipeline** |
| `q` | Quit without saving |
| Scroll wheel | Change slice |

### Stage 3 — Frangi vesselness filtering (~120 s)

Computes a multi-scale Frangi vesselness filter on a ROI-cropped volume around the seed points. Uses MPS (Metal Performance Shaders) on Apple Silicon if PyTorch is installed, otherwise falls back to CPU via scikit-image.

### Stage 4 — Per-vessel processing (LAD, LCX, RCA)

For each vessel the pipeline automatically:

1. Extracts the centerline from the vesselness map via FMM/Dijkstra shortest path
2. Clips to the proximal segment (LAD/LCX: 5–45 mm, RCA: 10–50 mm)
3. Estimates vessel radii along the centerline via EDT
4. Builds a tubular VOI (outer radius = 3× lumen radius)
5. Computes FAI statistics (mean HU, fat fraction, voxel count)
6. Generates CPR images (FAI overlay, vessel wall overlay, DICOM secondary capture)
7. Plots HU histogram and radial HU profile

### Stage 5 — Coronary Artery Contour Editor

Opens after all vessels are processed. Shows 3 MPR planes with vessel centerlines and wall contours overlaid.

**Vessel colors:** LAD = red · LCX = blue · RCA = green

| Key / Action | Effect |
|---|---|
| `1` / `2` / `3` | Switch active vessel |
| Click + drag a centerline point | Reposition that point; vessel mask updates on release |
| `[` / `]` | Decrease / increase radius at current point by 0.1 mm |
| `a` | Apply current radius to all points of the active vessel |
| `↑` / `↓` | Navigate slices |
| `←` / `→` | Navigate to next / previous centerline point |
| `p` or "Add PCAT" button | Generate PCAT volume (outer = radius × 3) with semi-transparent yellow overlay |
| `s` | **Save contours & PCAT mask, continue pipeline** |
| `q` | Quit without saving |
| Scroll wheel | Change slice |

> Every drag or radius change immediately rebuilds the vessel mask, so the overlay always reflects your edits.

### Stage 6 — CPR Browser (opens once per vessel)

After the contour editor closes, an interactive CPR browser opens for each vessel sequentially.

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
| `s` | Save a PNG snapshot |
| `q` | Close and continue to next vessel |

The cross-section shows:

- Actual vessel lumen contour (cyan, detected from HU thresholding)
- VOI boundary ring (green dashed, 3× lumen radius)

### Stage 7 — Export & visualization

1. Combined all-vessel VOI exported as `.raw` file
2. 3D visualization rendered as DICOM secondary capture frames
3. Summary bar chart and `*_results.json` saved

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
│   ├── patient1200_LAD_hu_histogram.png    # HU distribution in VOI
│   ├── patient1200_LAD_radial_profile.png  # Fat HU vs. distance from wall
│   └── patient1200_summary.png             # Bar charts: FAI, fat fraction, voxel count
├── raw/
│   ├── patient1200_LAD_voi.raw             # Per-vessel VOI volume
│   ├── patient1200_LAD_voi_metadata.json   # Volume dimensions, spacing, origin
│   ├── patient1200_combined_voi.raw        # All-vessel combined VOI
│   └── patient1200_combined_voi_metadata.json
└── 3d/
    └── patient1200_3d_dicom/               # 3D render as multi-frame DICOM
        ├── patient1200_3d_frame001.dcm
        └── ...
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
| `pipeline/coronary_contour_editor.py` | Edit contours: `python pipeline/coronary_contour_editor.py --dicom <dir> --data <npz> --output <dir>` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "too few centerline points" | Re-run and adjust seeds in the Seed Reviewer |
| NaN mean HU in results | No fat voxels in VOI; vessel may be heavily calcified |
| "only N vessels found" warning | Add the missing vessel manually in the Seed Reviewer |
| Seed Reviewer / Contour Editor won't open | Add `--skip-editor --skip-cpr-browser` (headless server) |
| TotalSegmentator fails | Place seeds manually with `seed_picker.py` (see Standalone Tools) |
| Frangi is slow (~120 s on CPU) | Install PyTorch (`pip install torch`) for MPS GPU acceleration on Apple Silicon |

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
