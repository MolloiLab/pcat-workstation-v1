# PCAT — Pericoronary Adipose Tissue Segmentation Pipeline

Automated Python pipeline for PCAT/PVAT segmentation from CCTA (Cardiac CT Angiography) images.

---

## Overview

The pipeline runs in **two modes**:

| Mode | Seeds | VOI Review | CPR Review | Use Case |
|---|---|---|---|---|
| **Fully Automatic** | TotalSegmentator (auto) | Skipped | Skipped | Batch / headless / research |
| **Semi-Automatic (Clinical QA)** | TotalSegmentator (auto) | Interactive editor | Interactive CPR browser | Clinical reporting |

Both modes run the same core algorithm — DICOM loading → Frangi vesselness → centerline extraction → tubular VOI construction → PCAT statistics → export + visualizations. The difference is whether manual review steps are presented to the operator.

### Core Modules

| Module | Role |
|---|---|
| `dicom_loader.py` | Load DICOM series → (Z,Y,X) HU volume |
| `auto_seeds.py` | TotalSegmentator wrapper — detects coronary ostia + waypoints automatically |
| `centerline.py` | Frangi vesselness + Fast Marching / Dijkstra centerline extraction |
| `pcat_segment.py` | Tubular VOI construction + PCAT HU filtering |
| `export_raw.py` | Export VOI as `.raw` binary + `metadata.json` |
| `visualize.py` | CPR (FAI, grayscale, native), HU histogram, radial profile, 3D render, summary chart |
| `run_pipeline.py` | Pipeline orchestrator (single patient + batch) |

### Clinical QA Tools (Semi-Automatic mode only)

| Tool | Launched By | Purpose |
|---|---|---|
| `seed_reviewer.py` | Standalone | Review / correct auto-generated seeds before running the pipeline |
| `voi_editor.py` | Auto-launched by pipeline | Interactive MPR brush editor — clinician reviews the auto-segmented VOI |
| `cpr_browser.py` | Auto-launched by pipeline | Interactive CPR viewer with live cross-section panel |
| `vessel_wall_editor.py` | Standalone | Adjust inner/outer vessel wall radii if CPR or radial profile shows boundary errors |
| `seed_picker.py` | Standalone | Manual seed placement (fallback when TotalSegmentator fails) |

---

## Clinical Background

PCAT refers to the fat tissue immediately surrounding coronary arteries. The **Fat Attenuation Index (FAI)** is the mean HU of PCAT fat voxels in the −190 to −30 HU range within the pericoronary VOI. A mean FAI above −70.1 HU indicates elevated pericoronary inflammation and high cardiovascular risk, as established by Oikonomou et al. in the CRISP-CT trial (Lancet 2018).

This pipeline segments the proximal 40 mm of the LAD and LCX arteries and the proximal 10–50 mm of the RCA for PCAT analysis.

---

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- TotalSegmentator licence (free for research — register at `https://backend.totalsegmentator.com/license-academic/`)
- `pyvista` (optional — required only for 3D render; `pip install pyvista`)

---

## Installation

```bash
git clone https://github.com/MolloiLab/PCAT.git
cd PCAT
pip install -r requirements.txt
```

---

## Mode 1 — Fully Automatic

Zero interaction. Suitable for batch processing, CI, and headless servers.

> **Research / bulk processing only.** This mode skips the mandatory clinical VOI review. Do not use for clinical reporting without a separate manual review step.

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds \
    --skip-editor \
    --skip-cpr-browser
```

**Batch:**
```bash
python pipeline/run_pipeline.py --batch --auto-seeds --skip-editor --skip-cpr-browser
```

**What it does:**
1. Runs TotalSegmentator to locate coronary ostia + waypoints
2. Computes Frangi vesselness (ROI-cropped, ~10 s)
3. Extracts centerlines, builds tubular VOIs, computes FAI statistics
4. Exports per-vessel `.raw` binary masks + `metadata.json`
5. Generates CPR images, HU histograms, radial profiles, 3D render, and summary chart

---

## Mode 2 — Semi-Automatic (Clinical QA Workflow)

Recommended for clinical reporting. The same automated pipeline runs, but pauses for mandatory operator review at the VOI and CPR stages.

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds
```

**Batch:**
```bash
python pipeline/run_pipeline.py --batch --auto-seeds
```

**What it does (same as above, plus):**
- **For each vessel**: opens the **VOI Editor** so the radiologist/cardiologist can review and approve the auto-segmented VOI before proceeding
- **For each vessel**: opens the **CPR Browser** to inspect cross-sections along the full vessel length

### Optional: Review auto-seeds before running the pipeline

If TotalSegmentator seeds look unreliable (e.g., only 2 of 3 vessels detected), review them first:

```bash
python pipeline/seed_reviewer.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/1200_2_auto.json \
    --output seeds/patient_1200_reviewed.json
```

Then pass the reviewed seeds file to the pipeline with `--seeds seeds/patient_1200_reviewed.json`.

---

## Clinical QA Tools — Reference

### Seed Reviewer (`seed_reviewer.py`)

Use after `--auto-seeds` to verify vessel detection before running the pipeline.

```bash
python pipeline/seed_reviewer.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/1200_2_auto.json \
    --output seeds/patient_1200_reviewed.json
```

**Confidence badges:**
- ✓ GREEN — high confidence, no issues detected
- ⚠ YELLOW — review recommended (fallback detection, sub-voxel radius, etc.)
- ✗ RED — critical issue: missing vessel, must correct before running pipeline

**Controls:** `1`/`2`/`3` switch vessel · click moves/adds seed · `d` deletes waypoint · `c` clears badges · `s` saves

---

### VOI Editor (`voi_editor.py`)

Auto-launched by the pipeline in Clinical QA mode. Can also be run standalone to re-review a saved mask.

```bash
python pipeline/voi_editor.py \
    --dicom Rahaf_Patients/1200.2 \
    --voi output/patient_1200/patient1200_LAD_voi_reviewed.npy \
    --vessel LAD \
    --output output/patient_1200/patient1200_LAD_voi_corrected.npy
```

Shows all three MPR views (axial, coronal, sagittal). The auto-segmented VOI is overlaid in **blood-red**; once edited it switches to **yellow-gold**.

| Toolbar Button | Key | Effect |
|---|---|---|
| 🖌 Paint | `1` | Left-drag adds voxels to VOI |
| ⌫ Erase | `2` | Left-drag removes voxels from VOI |
| ⭕ Fat-only | `3` | Left-drag adds only fat voxels (−190 to −30 HU) |
| ✋ Pan | `4` | Left-drag pans the view |

**Additional controls:**

| Key / Action | Effect |
|---|---|
| Right-click drag | Always erases |
| `[` / `]` or `+` / `-` | Decrease / increase brush radius (1–20 voxels) |
| `Ctrl+scroll` | Zoom in / out |
| `u` or `Ctrl+Z` | Undo last stroke (up to 30 levels) |
| `r` | Reset zoom |
| `s` | Save mask |
| `q` | Quit without saving |
| Scroll wheel | Change slice |
| `w` / `W` | Window width wider / narrower |
| `l` / `L` | Window level up / down |

**Output:** `<prefix>_<VESSEL>_voi_reviewed.npy` in the output directory.

---

### CPR Browser (`cpr_browser.py`)

Auto-launched by the pipeline in Clinical QA mode. Can also be run standalone.

```bash
python pipeline/cpr_browser.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200.json \
    --vessel LAD

# With a reviewed VOI mask:
python pipeline/cpr_browser.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200.json \
    --vessel LAD \
    --voi output/patient_1200/patient1200_LAD_voi_reviewed.npy
```

**Left panel (CPR):** Vessel runs top → bottom (ostium at top). Cyan horizontal line = "needle" position.  
**Right panel (cross-section):** Orthogonal slice at the needle position. White circle = lumen boundary; yellow dashed circle = PCAT VOI outer boundary; yellow/red overlay = fat voxels.

| Key / Action | Effect |
|---|---|
| Click CPR panel | Jump needle to that arc-length position |
| Drag slider | Move needle smoothly |
| `←` / `→` | Step needle by one centerline point |
| `s` | Save PNG snapshot |
| `q` | Quit |

---

### Vessel Wall Editor (`vessel_wall_editor.py`)

Use if the radial profile or CPR shows boundary errors after the pipeline runs.

```bash
python pipeline/vessel_wall_editor.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200.json \
    --vessel LAD \
    --output output/patient_1200/LAD_wall_override.json
```

Red circle = lumen (inner wall). Yellow circle = PCAT boundary (outer wall).

| Key | Effect |
|---|---|
| Arrow keys | Navigate between vessel points |
| `[` / `]` | Decrease / increase inner radius |
| `{` / `}` | Decrease / increase outer radius |
| `a` / `A` | Apply current radii to all points |
| `s` | Save and exit |

---

### Manual Seed Picker (`seed_picker.py`)

Fallback for when TotalSegmentator cannot detect a vessel. Place seeds manually then run the pipeline with `--seeds`.

```bash
python pipeline/seed_picker.py \
    --dicom Rahaf_Patients/1200.2 \
    --output seeds/patient_1200.json
```

| Key / Action | Effect |
|---|---|
| `1` / `2` / `3` | Switch active vessel: LAD / LCX / RCA |
| Left-click | Add seed point (ostium first, then waypoints) |
| `u` | Undo last point |
| `r` | Reset current vessel |
| `s` | Save and continue |
| `q` | Save and quit |
| Scroll wheel | Change slice |
| `w` / `W` | Window width wider / narrower |
| `l` / `L` | Window level up / down |

---

## Output Files

Each patient run produces the following outputs per vessel (e.g. `patient1200_LAD_*`):

| File | Description |
|---|---|
| `<prefix>_<VESSEL>_voi.raw` | Binary VOI mask (int8, same shape as input volume) |
| `<prefix>_<VESSEL>_voi_metadata.json` | VOI metadata: shape, spacing, origin, affine |
| `<prefix>_combined_voi.raw` | Union of all vessel VOI masks |
| `<prefix>_combined_voi_metadata.json` | Combined VOI metadata |
| `<prefix>_<VESSEL>_voi_reviewed.npy` | Clinician-reviewed VOI mask (numpy bool, shape Z×Y×X) — Clinical QA mode only |
| `<prefix>_3d_voi.png` | 3D semi-transparent render of all PCAT VOIs + artery tubes |
| `<prefix>_<VESSEL>_cpr_fai.png` | CPR with FAI overlay — vessel top-to-bottom |
| `<prefix>_<VESSEL>_cpr_gray.png` | Grayscale CPR (multi-rotation) |
| `<prefix>_<VESSEL>_cpr_native_*.png` | Syngo.via-style curved CPR (3 rotation views) |
| `<prefix>_<VESSEL>_hu_histogram.png` | HU distribution histogram of PCAT fat voxels |
| `<prefix>_<VESSEL>_radial_profile.png` | Radial HU profile (1 mm rings, 0–20 mm from vessel wall) |
| `<prefix>_summary.png` | Summary bar charts: mean FAI HU, fat fraction, voxel count |
| `<prefix>_results.json` | Per-vessel FAI statistics |

> **Viewing `.raw` files:** Open in ImageJ using `File → Import → Raw`. Set the correct dimensions and data type from the accompanying `_metadata.json`. Alternatively, load in Python: `np.fromfile(path, dtype=np.int8).reshape(shape)`.

---

## Clinical Interpretation Guide

**CPR image:** The vessel centerline runs **vertically** — top = ostium, bottom = distal end. Yellow/red regions indicate PCAT fat tissue. The dashed white vertical line marks the vessel axis. Focus review on pericoronary fat signal to the left and right of the centerline.

**HU Histogram:** Orange bars = fat voxels. The dashed vertical line at −70.1 HU = FAI risk threshold. Mean FAI above −70.1 HU indicates elevated cardiovascular inflammation risk.

**Radial Profile:** Mean HU of fat voxels at each radial distance from the vessel wall. Y-axis: −90 to −65 HU (Antonopoulos et al. 2018 clinical range). Values closer to −65 HU (less negative) = more inflamed pericoronary fat.

**FAI Risk Threshold:** −70.1 HU (Oikonomou 2018). Above this threshold (less negative) = higher pericoronary inflammation risk.

---

## PCAT Parameters

| Parameter | Value | Source |
|---|---|---|
| FAI HU range | −190 to −30 HU | Antonopoulos et al. 2017 |
| FAI risk threshold | −70.1 HU | Oikonomou et al. 2018 (CRISP-CT) |
| LAD segment | Proximal 0–40 mm | Standard protocol |
| LCX segment | Proximal 0–40 mm | Standard protocol |
| RCA segment | Proximal 10–50 mm | Standard protocol |
| VOI thickness | Mean vessel diameter (1×) from outer wall | Iacobellis et al. |
| Radial profile y-axis | −90 to −65 HU | Antonopoulos et al. 2018 |

---

## Seed JSON Format

```json
{
  "LAD": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [[z1, y1, x1], [z2, y2, x2]],
    "segment_length_mm": 40.0
  },
  "LCX": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [[z1, y1, x1]],
    "segment_length_mm": 40.0
  },
  "RCA": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [[z1, y1, x1], [z2, y2, x2]],
    "segment_start_mm": 10.0,
    "segment_length_mm": 50.0
  }
}
```

All coordinates are **IJK voxel indices** `[z, y, x]` (0-indexed) in the loaded volume.

---

## CLI Reference

```
python pipeline/run_pipeline.py [--batch | --dicom PATH]

Required (single patient mode):
  --dicom PATH             Path to DICOM series directory

Seed source (one required):
  --seeds PATH             Path to seed JSON file
  --auto-seeds             Run TotalSegmentator to generate seeds automatically

Options:
  --output DIR             Output directory (default: output/patient)
  --prefix STR             Filename prefix for all outputs (default: pcat)
  --vessels LAD LCX RCA   Vessels to process (default: all in seeds file)
  --skip-3d                Skip 3D pyvista rendering (headless environments)
  --skip-editor            Skip interactive VOI editor (headless / fully automatic mode)
  --skip-cpr-browser       Skip interactive CPR browser (headless / fully automatic mode)
  --batch                  Run all patients from PATIENT_CONFIGS in run_pipeline.py
  --project-root DIR       Base directory for batch mode relative paths (default: .)
  --auto-seeds-device      Device for TotalSegmentator: cpu | gpu | mps (auto-detected)
  --auto-seeds-license KEY TotalSegmentator academic licence key
```

---

## Changing Input DICOM Data

**Single patient:** Change the `--dicom` path to any DICOM series folder.

**Batch mode:** Edit the `PATIENT_CONFIGS` list in `run_pipeline.py` (lines ~88–110). Each entry requires `patient_id`, `dicom`, `seeds`, `output`, and `prefix`.

```python
PATIENT_CONFIGS = [
    {"patient_id": "1200", "dicom": "Rahaf_Patients/1200.2", "seeds": "seeds/patient_1200.json", "output": "output/patient_1200", "prefix": "patient1200"},
    # Add new patient:
    {"patient_id": "999",  "dicom": "New_Patients/patient999", "seeds": "seeds/patient_999.json", "output": "output/patient_999", "prefix": "patient999"},
]
```

**Spacing:** Auto-detected from DICOM tags. No manual configuration needed.

**Non-Siemens data:** `dicom_loader.py` handles non-standard `RescaleIntercept` values automatically via standard DICOM tags.

---

## Patient Data

| Patient | DICOM Dir | Slices | Shape | Spacing (z×y×x mm) |
|---|---|---|---|---|
| 1200 | `Rahaf_Patients/1200.2/` | 405 | 512×512×405 | 0.500×0.3241×0.3241 |
| 2 | `Rahaf_Patients/2.1/` | 149 | 512×512×149 | 0.500×0.3758×0.3758 |
| 317 | `Rahaf_Patients/317.6/` | 399 | 512×512×399 | 0.500×0.3388×0.3388 |

> **Note:** All scans use Siemens syngo.via with non-standard `RescaleIntercept = −8192`. Handled automatically.

> **Note:** Patient 2 has only 149 slices (~74.5 mm Z-span). Verify that the proximal coronary segments fall within the scan range before running.

---

## Project Structure

```
PCAT/
├── pipeline/
│   ├── __init__.py
│   ├── dicom_loader.py       # DICOM I/O + HU conversion
│   ├── auto_seeds.py         # TotalSegmentator wrapper for automatic seed generation
│   ├── centerline.py         # Frangi vesselness + Fast Marching centerline extraction
│   ├── pcat_segment.py       # Tubular VOI + FAI HU filtering
│   ├── export_raw.py         # .raw binary + metadata.json export
│   ├── visualize.py          # All visualization outputs (CPR, histogram, profile, 3D, summary)
│   ├── seed_picker.py        # Interactive MPR seed picker (manual fallback)
│   ├── seed_reviewer.py      # Interactive GUI for reviewing auto-generated seeds
│   ├── voi_editor.py         # Interactive MPR VOI brush editor
│   ├── cpr_browser.py        # Interactive CPR browser with live cross-section panel
│   ├── vessel_wall_editor.py # Interactive GUI for adjusting vessel wall radii
│   └── run_pipeline.py       # Pipeline orchestrator (single + batch)
├── seeds/
│   ├── patient_1200.json
│   ├── patient_2.json
│   └── patient_317.json
├── docs/
│   └── literature_review.md
├── output/                   # Generated outputs (gitignored)
├── Rahaf_Patients/           # DICOM data (gitignored)
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| "too few centerline points" | Seed outside vessel, or vessel too short | Re-run seed_picker or seed_reviewer |
| NaN mean HU in stats JSON | No fat voxels (−190 to −30 HU) in VOI | Check VOI size; vessel may be calcified |
| "only N vessels found" warning | TotalSegmentator missed a vessel | Run seed_reviewer, fix missing vessel manually |
| 3D render skipped | pyvista not installed | `pip install pyvista` |
| Short centerline (< 10 mm) | Waypoints too far from vessel or bad seed | Add more waypoints in seed_picker |
| Sub-voxel radius warning | Vessel too small relative to DICOM resolution | Review in vessel_wall_editor, increase min radius |
| VOI editor won't open | Headless server (no display) | Use `--skip-editor` and review outputs manually |
| CPR browser won't open | Headless or TkAgg not available | Use `--skip-cpr-browser`; run `cpr_browser.py` separately |
