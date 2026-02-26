# PCAT — Pericoronary Adipose Tissue Segmentation Pipeline

Automated Python pipeline for PCAT/PVAT segmentation from CCTA (Cardiac CT Angiography) images.

---

## Overview

| Module | Step | Description |
|---|---|---|
| `dicom_loader.py` | 1 | Load DICOM series → (Z,Y,X) HU volume |
| `seed_picker.py` | 2a | Interactive seed picker (ostium + waypoints) |
| `seed_reviewer.py` | 2b | Interactive GUI for reviewing and correcting auto-generated seeds with clinical warning badges |
| `centerline.py` | 3 | Frangi vesselness + Dijkstra centerline extraction |
| `pcat_segment.py` | 4 | Tubular VOI construction + PCAT HU filtering |
| `voi_editor.py` | 5 | **Mandatory** interactive MPR brush editor — clinician reviews and edits the auto-segmented VOI |
| `vessel_wall_editor.py` | 6 | Interactive GUI for reviewing/adjusting inner and outer vessel wall radii per-point |
| `export_raw.py` | 7 | Export VOI as `.nii.gz` binary NIfTI mask |
| `visualize.py` | 8 | 5 analysis outputs (3D render, CPR, histogram, radial profile, summary) |
| `run_pipeline.py` | — | Orchestrates all steps for single or batch patients |

---

## Clinical Background

PCAT refers to the fat tissue immediately surrounding coronary arteries. The Fat Attenuation Index (FAI) is calculated as the mean Hounsfield Unit (HU) value of PCAT fat voxels within the −190 to −30 HU range. A mean FAI value greater than −70.1 HU indicates high cardiovascular inflammation risk, as established by Oikonomou et al. in the 2018 CRISP-CT trial. This pipeline segments the proximal 40 mm of the LAD and LCX arteries and the proximal 10–50 mm of the RCA for PCAT analysis.

---

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- TotalSegmentator license (free for research use, requires `totalseg_setup_weights`)
- `pyvista` (optional, for 3D render functionality)

---

## Installation

```bash
git clone https://github.com/MolloiLab/PCAT.git
cd PCAT
pip install -r requirements.txt
```

---

## Quick Start — Clinical QA Workflow (default / recommended)

The standard workflow: auto-generate seeds, run the pipeline, and use the
interactive review tools to verify each vessel.

```bash
# Step 1: Run pipeline with auto-generated seeds
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds
```
This will:
1. Run TotalSegmentator to locate coronary artery seed points automatically
2. Extract centerlines, build tubular VOIs, and compute FAI statistics
3. **For each vessel, open the mandatory VOI Editor** so the radiologist/cardiologist can review and approve the auto-segmented VOI
4. **For each vessel, open the interactive CPR Browser** so you can inspect cross-sections along the vessel
5. Export per-vessel `.nii.gz` binary masks and visualization figures

```bash
# Step 2: Review auto-generated seeds (optional but recommended)
python pipeline/seed_reviewer.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/1200_2_auto.json \
    --output seeds/patient_1200_reviewed.json
```

For processing multiple patients in batch mode:

```bash
python pipeline/run_pipeline.py --batch --auto-seeds
```

**Headless / CI environments** (no display available): add `--skip-editor --skip-cpr-browser` to bypass all interactive review steps:

```bash
python pipeline/run_pipeline.py --dicom ... --auto-seeds --skip-editor --skip-cpr-browser
```

> **Warning:** `--skip-editor` bypasses the mandatory clinical review. Only use this when a separate manual review will be performed after the fact.

---

## Quick Start — Fully Automatic Mode

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds \
    --skip-editor \
    --skip-cpr-browser
```

This runs the full pipeline without any interactive windows — suitable for batch/headless/CI environments.

## Manual Seed Workflow

### Step A: Seed Picker

Launch the interactive seed picker to identify vessel ostia and waypoints:

```bash
python pipeline/seed_picker.py \
    --dicom Rahaf_Patients/1200.2 \
    --output seeds/patient_1200.json
```

**Seed Picker Controls:**

| Key / Action | Effect |
|---|---|
| `1` / `2` / `3` | Switch active vessel: LAD / LCX / RCA |
| Left-click | Add seed point (ostium first, then waypoints) |
| `u` | Undo last point |
| `r` | Reset current vessel |
| `s` | Save and continue |
| `q` | Save and quit |
| Scroll wheel | Change slice in each MPR panel |
| `w` / `W` | Window width wider / narrower |
| `l` / `L` | Window level up / down |

Repeat for each patient:
```bash
python pipeline/seed_picker.py --dicom Rahaf_Patients/2.1    --output seeds/patient_2.json
python pipeline/seed_picker.py --dicom Rahaf_Patients/317.6  --output seeds/patient_317.json
```

### Step B: Run Pipeline with Seeds

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200.json \
    --output output/patient_1200 \
    --prefix patient1200
```

---

## Changing Input DICOM Data

For processing new patient data:

**Single patient:** Simply change the `--dicom` path to point to any DICOM series folder.

**Batch mode:** Edit the `PATIENT_CONFIGS` list in `run_pipeline.py` (lines ~84–106). Each entry requires:
- `patient_id`: Short string identifier
- `dicom`: Path to DICOM folder (relative to `--project-root`)
- `seeds`: Path to seed JSON file
- `output`: Output directory path
- `prefix`: Output file prefix

Example of adding a fourth patient:
```python
PATIENT_CONFIGS = [
    {"patient_id": "1200", "dicom": "Rahaf_Patients/1200.2", "seeds": "seeds/patient_1200.json", "output": "output/patient_1200", "prefix": "patient1200"},
    {"patient_id": "2",    "dicom": "Rahaf_Patients/2.1",    "seeds": "seeds/patient_2.json",    "output": "output/patient_2",    "prefix": "patient2"},
    {"patient_id": "317",  "dicom": "Rahaf_Patients/317.6",  "seeds": "seeds/patient_317.json",  "output": "output/patient_317", "prefix": "patient317"},
    # Add new patient:
    {"patient_id": "999",  "dicom": "New_Patients/patient999", "seeds": "seeds/patient_999.json", "output": "output/patient_999", "prefix": "patient999"},
]
```

**Custom spacing:** Spacing is auto-detected from DICOM tags. No manual configuration needed.

**Non-Siemens data:** If `RescaleIntercept` is not `−8192`, `dicom_loader.py` handles it automatically via standard DICOM tags.

---

## Clinical QA Workflow — Manual Review Tools

### Step 1: Seed Reviewer (`seed_reviewer.py`)

Use this tool after running `--auto-seeds` to verify that TotalSegmentator correctly identified all vessels:

```bash
python pipeline/seed_reviewer.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/1200_2_auto.json \
    --output seeds/patient_1200_reviewed.json
```

**Color badges indicate confidence level:**
- ✓ GREEN = high confidence, no issues detected
- ⚠ YELLOW = review recommended (fallback detection, sub-voxel radius, etc.)
- ✗ RED = critical issue: missing vessel, must correct before running pipeline

**Keyboard controls:**
- `1` / `2` / `3` = Switch between LAD, LCX, RCA
- Click = Move existing seed or add new seed
- `d` = Delete waypoint
- `c` = Clear warning badges
- `s` = Save and exit

---

### Step 2: VOI Editor — Mandatory Sanity Check (`voi_editor.py`)

After the pipeline builds the tubular VOI for each vessel, it automatically launches the VOI editor. This is a **mandatory clinical review step** — the pipeline blocks until the window is closed.

The editor shows all three MPR views (axial, coronal, sagittal) with the auto-segmented VOI overlaid in semi-transparent yellow. The clinician can paint voxels on or off before saving.

```bash
# Standalone usage (e.g. to re-review a saved mask):
python pipeline/voi_editor.py \
    --dicom Rahaf_Patients/1200.2 \
    --voi output/patient_1200/patient1200_LAD_voi_reviewed.npy \
    --vessel LAD \
    --output output/patient_1200/patient1200_LAD_voi_corrected.npy
```

**VOI Editor Controls:**

| Key / Action | Effect |
|---|---|
| Left-click drag | ADD voxels to VOI mask |
| Right-click drag | REMOVE voxels from VOI mask |
| `+` / `-` | Increase / decrease brush radius (default: 2 voxels) |
| `u` or `Ctrl+Z` | Undo last stroke (up to 20 levels) |
| `s` | Save edited mask as `.npy` (and `.nii.gz` if nibabel available) |
| `q` | Quit without saving |
| Scroll wheel | Change slice in each MPR panel |
| `w` / `W` | Window width wider / narrower |
| `l` / `L` | Window level up / down |

**Orientation:** All three views use anatomically correct orientation — head at top, feet at bottom (the upside-down orientation bug from older seed picker versions is fixed here).

**Output:** The reviewed mask is saved as `<prefix>_<VESSEL>_voi_reviewed.npy` in the output directory. If `nibabel` is installed, a `.nii.gz` file is also saved alongside it.

---

### Step 3: CPR Browser — Interactive Cross-Section Review (`cpr_browser.py`)

After the pipeline generates the CPR images, the CPR browser auto-launches for each vessel. You can also run it standalone:

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

**CPR Browser Controls:**

| Key / Action | Effect |
|---|---|
| Click CPR panel | Jump needle to that arc-length position |
| Drag slider | Move needle smoothly along the vessel |
| `←` / `→` arrow keys | Step needle by one centerline point |
| `s` | Save PNG snapshot of current view |
| `q` | Quit |

**Left panel (CPR):** The vessel runs top→bottom (ostium at top). The cyan horizontal line is the "needle" — it marks the position shown in the right panel.

**Right panel (cross-section):** Orthogonal slice through the vessel at the needle position.
- White circle = vessel lumen boundary (radius from EDT estimate)
- Yellow dashed circle = PCAT VOI outer boundary (2× lumen radius)
- Yellow/red overlay = FAI fat voxels (−190 to −30 HU)

---

### Step 4: Vessel Wall Editor (`vessel_wall_editor.py`)

Use this tool after pipeline runs if the radial profile or CPR shows boundary errors:

```bash
python pipeline/vessel_wall_editor.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200.json \
    --vessel LAD \
    --output output/patient_1200/LAD_wall_override.json
```

The interface shows:
- Red circle = lumen (inner vessel wall)
- Yellow circle = PCAT boundary (outer vessel wall)

**Keyboard controls:**
- Arrow keys = Navigate between vessel points
- `[` / `]` = Decrease/increase inner radius
- `{` / `}` = Decrease/increase outer radius
- `a` / `A` = Apply current radii to all points
- `s` = Save and exit

---

## Output Files

Each patient run produces the following outputs for each vessel (e.g. `patient1200_LAD_*`):

| File | Description |
|---|---|
| `<prefix>_<VESSEL>_voi.nii.gz` | Binary VOI mask (int8, `1`=VOI, `0`=outside) in NIfTI format |
| `<prefix>_combined_voi.nii.gz` | Union of all vessel VOI masks (single NIfTI file) |
| `<prefix>_<VESSEL>_voi_reviewed.npy` | Clinician-reviewed VOI mask (numpy bool array, shape Z×Y×X) |
| `<prefix>_3d_voi.png` | 3D semi-transparent render of all PCAT VOIs + artery tubes |
| `<prefix>_<VESSEL>_cpr_fai.png` | Bishop-frame CPR with FAI overlay (vertical orientation) |
| `cpr_browser_<VESSEL>_<N>mm.png` | Snapshot from interactive CPR browser at arc-length N mm |
| `<prefix>_<VESSEL>_hu_histogram.png` | HU distribution histogram of PCAT fat voxels |
| `<prefix>_<VESSEL>_radial_profile.png` | Radial HU profile (1 mm rings, 0–20 mm from vessel wall) |
| `<prefix>_summary.png` | Summary bar charts: mean FAI HU, fat fraction, voxel count |
| `<prefix>_results.json` | Per-vessel FAI statistics (JSON) |

> **NIfTI viewing:** The `.nii.gz` files can be opened directly in ITK-SNAP, 3D Slicer, FSLeyes, or any NIfTI viewer. They can also be loaded in Python via `nibabel.load(path).get_fdata()`.

---

## Clinical Interpretation Guide

**CPR image:** The vessel centerline runs **vertically** — the top of the image is the ostium (start of the proximal segment) and the bottom is the distal end. The x-axis shows lateral distance from the centerline. Yellow/red regions indicate PCAT fat tissue. The dashed white vertical line marks the vessel centerline axis. Focus your review on the pericoronary fat signal to the left and right of the centerline.

**HU Histogram:** Orange bars represent fat voxels. The vertical dashed line at −70.1 HU indicates the FAI risk threshold. If the mean FAI HU is greater than −70.1, this suggests elevated cardiovascular inflammation risk.

**Radial Profile:** Each point represents the mean HU of fat voxels at that specific radial distance from the vessel wall. The y-axis is displayed in the clinical range of −90 to −65 HU, matching the typical FAI range reported in Antonopoulos et al. 2018 (*"Detecting human coronary inflammation by imaging perivascular fat"*). Values closer to −65 HU (less negative) indicate more inflamed pericoronary fat.

**FAI Risk Threshold:** The clinical threshold is −70.1 HU (Oikonomou 2018). Values above this threshold (less negative) indicate higher pericoronary inflammation.

---

## PCAT Parameters

| Parameter | Value | Source |
|---|---|---|
| FAI HU range | −190 to −30 HU | Antonopoulos et al. 2017 |
| FAI risk threshold | −70.1 HU | Oikonomou et al. 2018 (CRISP-CT) |
| LAD segment | Proximal 0–40 mm | Standard protocol |
| LCX segment | Proximal 0–40 mm | Standard protocol |
| RCA segment | Proximal 10–50 mm | Standard protocol |
| VOI thickness | Mean vessel diameter (1×) | Iacobellis et al. |
| Radial profile y-axis | −90 to −65 HU | Antonopoulos et al. 2018 |

---

## Seed JSON Format

After running `seed_picker.py`, seed files are auto-populated. The format is:

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

## Patient Data

| Patient | DICOM Dir | Slices | Shape | Spacing (z×y×x mm) |
|---|---|---|---|---|
| 1200 | `Rahaf_Patients/1200.2/` | 405 | 512×512×405 | 0.500×0.3241×0.3241 |
| 2 | `Rahaf_Patients/2.1/` | 149 | 512×512×149 | 0.500×0.3758×0.3758 |
| 317 | `Rahaf_Patients/317.6/` | 399 | 512×512×399 | 0.500×0.3388×0.3388 |

> **Note:** All scans use Siemens syngo.via with non-standard `RescaleIntercept = −8192`. This is handled automatically in `dicom_loader.py`.

> **Note:** Patient 2 has only 149 slices (~74.5 mm Z-span). Verify that the proximal coronary segments fall within the scan range before running.

---

## Project Structure

```
PCAT/
├── pipeline/
│   ├── __init__.py
│   ├── dicom_loader.py       # DICOM I/O + HU conversion
│   ├── centerline.py         # Frangi vesselness + Dijkstra centerline extraction
│   ├── pcat_segment.py       # Tubular VOI + FAI HU filtering
│   ├── export_raw.py         # NIfTI .nii.gz export (+ deprecated .raw for backward compat)
│   ├── visualize.py          # All 5 analysis output figures
│   ├── seed_picker.py        # Interactive MPR seed picker GUI
│   ├── seed_reviewer.py      # Interactive GUI for reviewing auto-generated seeds
│   ├── voi_editor.py         # Interactive MPR VOI brush editor (mandatory sanity check)
│   ├── cpr_browser.py        # Interactive CPR browser with live cross-section panel
│   ├── vessel_wall_editor.py # Interactive GUI for adjusting vessel wall radii
│   └── run_pipeline.py       # Pipeline orchestrator (single + batch)
├── seeds/
│   ├── patient_1200.json     # Seed coords for patient 1200 (fill via seed_picker)
│   ├── patient_2.json        # Seed coords for patient 2
│   └── patient_317.json      # Seed coords for patient 317
├── output/                   # Generated outputs (gitignored)
├── Rahaf_Patients/           # DICOM data (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## CLI Reference

```
python pipeline/run_pipeline.py [--batch | --dicom PATH]

Required (single patient mode):
  --dicom PATH            Path to DICOM series directory
  --seeds PATH            Path to seed JSON (required unless --auto-seeds)

Options:
  --output DIR            Output directory (default: output/patient)
  --prefix STR            Filename prefix for all outputs (default: pcat)
  --vessels LAD LCX RCA  Vessels to process (default: all in seeds file)
  --auto-seeds            Run TotalSegmentator to generate seeds automatically
  --auto-seeds-device     Device for TotalSegmentator: cpu | gpu | mps
  --auto-seeds-license    TotalSegmentator academic licence key
  --skip-3d               Skip 3D pyvista rendering (headless environments)
  --skip-editor           Skip the mandatory VOI editor review (headless/CI)
  --skip-cpr-browser      Skip the interactive CPR browser (headless/CI)
  --batch                 Run all patients from PATIENT_CONFIGS in run_pipeline.py
  --project-root DIR      Base directory for batch mode relative paths (default: .)
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| "too few centerline points" | Seed outside vessel, or vessel too short | Re-run seed_picker or seed_reviewer |
| NaN mean HU in stats JSON | No fat voxels (−190 to −30 HU) found in VOI | Check VOI size; vessel may be calcified |
| "only N vessels found" warning | TotalSegmentator couldn't find all 3 vessels | Run seed_reviewer, manually fix missing vessel |
| 3D render skipped | pyvista not installed | `pip install pyvista` |
| Short centerline (< 10 mm) | Waypoints too far from vessel or bad seed | Add more waypoints in seed_picker |
| Sub-voxel radius warning | Vessel too small relative to DICOM resolution | Review in vessel_wall_editor, increase min radius |
| VOI editor won't open | Running on a headless server (no display) | Use `--skip-editor` flag and review outputs manually |
| CPR browser won't open | Running headless or TkAgg not available | Use `--skip-cpr-browser`; run `cpr_browser.py` separately with display |
| `.nii.gz` export fails | nibabel not installed | `pip install nibabel` (already in requirements.txt) |
| VOI editor axes upside-down | Using seed_picker instead of voi_editor | voi_editor uses corrected orientation; seed_picker has known head-down issue |
