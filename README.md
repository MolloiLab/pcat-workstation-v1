# PCAT — Pericoronary Adipose Tissue Segmentation Pipeline

Automated Python pipeline for PCAT/PVAT segmentation from CCTA (Cardiac CT Angiography) images.

---

## Overview

| Module | Step | Description |
|---|---|---|
| `dicom_loader.py` | 1 | Load DICOM series → (Z,Y,X) HU volume |
| `seed_picker.py` | 2 | Interactive seed picker (ostium + waypoints) |
| `seed_reviewer.py` | 2 | Interactive GUI for reviewing and correcting auto-generated seeds with clinical warning badges |
| `centerline.py` | 3 | Frangi vesselness + Dijkstra centerline extraction |
| `pcat_segment.py` | 4 | Tubular VOI construction + PCAT HU filtering |
| `vessel_wall_editor.py` | 5 | Interactive GUI for reviewing/adjusting inner and outer vessel wall radii |
| `export_raw.py` | 6 | Export full-res VOI as `.raw` + metadata JSON |
| `visualize.py` | 7 | 5 analysis outputs (3D, CPR, histogram, radial profile) |
| `run_pipeline.py` | 8 | Orchestrates all steps for single or batch patients |

---

## Clinical Background

PCAT refers to the fat tissue immediately surrounding coronary arteries. The Fat Attenuation Index (FAI) is calculated as the mean Hounsfield Unit (HU) value of PCAT fat voxels within the -190 to -30 HU range. A mean FAI value greater than -70.1 HU indicates high cardiovascular inflammation risk, as established by Oikonomou et al. in the 2018 CRISP-CT trial. This pipeline segments the proximal 40mm of the LAD and LCX arteries and the proximal 10-50mm of the RCA for PCAT analysis.

---

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- TotalSegmentator license (free for research use, requires `totalseg_setup_weights`)
- pyvista (optional, for 3D render functionality)

---

## Installation

```bash
git clone https://github.com/your-repo/PCAT.git
cd PCAT
pip install -r requirements.txt
```

---

## Quick Start — Fully Automatic Mode (recommended)

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds
```

The `--auto-seeds` flag runs TotalSegmentator automatically, eliminating the need for manual seed picking. For processing multiple patients in batch mode:

```bash
python pipeline/run_pipeline.py --batch --auto-seeds
```

---

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

**Batch mode:** Edit the `BATCH_CONFIGS` list in `run_pipeline.py` (lines ~82-105). Each entry requires:
- `dicom_dir`: Path to DICOM folder
- `seeds_file`: Path to seed JSON file
- `output_dir`: Output directory path
- `prefix`: Output file prefix

Example of adding a fourth patient:
```python
BATCH_CONFIGS = [
    {"dicom_dir": "Rahaf_Patients/1200.2", "seeds_file": "seeds/patient_1200.json", "output_dir": "output/patient_1200", "prefix": "patient1200"},
    {"dicom_dir": "Rahaf_Patients/2.1", "seeds_file": "seeds/patient_2.json", "output_dir": "output/patient_2", "prefix": "patient2"},
    {"dicom_dir": "Rahaf_Patients/317.6", "seeds_file": "seeds/patient_317.json", "output_dir": "output/patient_317", "prefix": "patient317"},
    {"dicom_dir": "New_Patients/patient4", "seeds_file": "seeds/patient4.json", "output_dir": "output/patient4", "prefix": "patient4"},
]
```

**Custom spacing:** Spacing is auto-detected from DICOM tags. No manual configuration needed.

**Non-Siemens data:** If `RescaleIntercept` is not `-8192`, the `dicom_loader.py` handles it automatically via standard DICOM tags.

---

## Clinical QA Workflow — Manual Review Tools

### Seed Reviewer (`seed_reviewer.py`)

Use this tool after running `--auto-seeds` to verify that TotalSegmentator correctly identified all vessels:

```bash
python pipeline/seed_reviewer.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200_auto.json \
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

### Vessel Wall Editor (`vessel_wall_editor.py`)

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

Each patient run produces the following outputs for each vessel:

| File | Output # | Description |
|------|----------|-------------|
| `<prefix>_3d_voi.png` | 1 | 3D semi-transparent render of all PCAT VOIs + artery tubes |
| `<prefix>_<VESSEL>_voi.raw` | 2 | Binary VOI mask (int16, same dims as input CCTA) |
| `<prefix>_<VESSEL>_voi_meta.json` | 2 | .raw metadata: shape, spacing, dtype, voxel count |
| `<prefix>_<VESSEL>_cpr_fai.png` | 3 | Bishop-frame straightened CPR with FAI overlay |
| `<prefix>_<VESSEL>_hu_histogram.png` | 4 | HU distribution histogram of PCAT fat voxels |
| `<prefix>_<VESSEL>_radial_profile.png` | 5 | Radial HU profile (1mm rings, 0-20mm from vessel wall) |
| `<prefix>_summary.png` | — | Summary bar charts: mean FAI HU, fat fraction, voxel count |
| `<prefix>_stats.json` | — | Per-vessel FAI statistics (JSON) |

---

## Clinical Interpretation Guide

**CPR image (Output 3):** The x-axis represents arc-length along the vessel in millimeters. The y-axis shows lateral distance in millimeters. Yellow and red regions indicate fat tissue. The dashed white line marks the vessel centerline. Focus your review on the pericoronary fat signal above and below the centerline.

**HU Histogram (Output 4):** Orange bars represent fat voxels. The vertical dashed line at -70.1 HU indicates the FAI risk threshold. If the mean FAI HU is greater than -70.1, this suggests elevated cardiovascular inflammation risk.

**Radial Profile (Output 5):** Each point represents the mean HU of fat voxels at that specific distance from the vessel wall. This visualization helps determine if inflammation is close to the vessel (0-5mm) or located farther away.

**FAI Risk Threshold:** The clinical threshold is -70.1 HU (Oikonomou 2018). Values above this threshold (less negative) indicate higher pericoronary inflammation.

---

## PCAT Parameters

| Parameter | Value | Source |
|---|---|---|
| FAI HU range | −190 to −30 HU | Antonopoulos et al. 2017 |
| LAD segment | Proximal 0–40 mm | Standard protocol |
| LCX segment | Proximal 0–40 mm | Standard protocol |
| RCA segment | Proximal 10–50 mm | Standard protocol |
| VOI thickness | Mean vessel diameter (1×) | Iacobellis et al. |

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

> **Note**: All scans use Siemens syngo.via with non-standard `RescaleIntercept = -8192`. This is handled automatically in `dicom_loader.py`.

> **Note**: Patient 2 has only 149 slices (~74.5mm Z-span). Verify that the proximal coronary segments fall within the scan range before running.

---

## Project Structure

```
PCAT/
├── pipeline/
│   ├── __init__.py
│   ├── dicom_loader.py      # DICOM I/O + HU conversion
│   ├── centerline.py        # Frangi + Dijkstra centerline extraction
│   ├── pcat_segment.py      # Tubular VOI + FAI HU filtering
│   ├── export_raw.py        # .raw export (full-resolution)
│   ├── visualize.py         # All 5 analysis output figures
│   ├── seed_picker.py       # Interactive MPR seed picker GUI
│   ├── seed_reviewer.py     # Interactive GUI for reviewing auto-generated seeds
│   ├── vessel_wall_editor.py # Interactive GUI for adjusting vessel wall radii
│   └── run_pipeline.py      # Pipeline orchestrator (single + batch)
├── seeds/
│   ├── patient_1200.json    # Seed coords for patient 1200 (fill via seed_picker)
│   ├── patient_2.json       # Seed coords for patient 2
│   └── patient_317.json     # Seed coords for patient 317
├── output/                  # Generated outputs (gitignored)
├── Rahaf_Patients/          # DICOM data (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| "too few centerline points" | Seed outside vessel, or vessel too short | Re-run seed_picker or seed_reviewer |
| NaN mean HU in stats JSON | No fat voxels (-190 to -30 HU) found in VOI | Check VOI size; vessel may be calcified |
| "only N vessels found" warning | TotalSegmentator couldn't find all 3 vessels | Run seed_reviewer, manually fix missing vessel |
| 3D render skipped | pyvista not installed | pip install pyvista |
| Short centerline (< 10mm) | Waypoints too far from vessel or bad seed | Add more waypoints in seed_picker |
| Sub-voxel radius warning | Vessel too small relative to DICOM resolution | Review in vessel_wall_editor, increase min radius |