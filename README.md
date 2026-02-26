# PCAT — Pericoronary Adipose Tissue Segmentation Pipeline

Automated Python pipeline for PCAT/PVAT segmentation from CCTA images.  
Extracts proximal coronary artery segments (LAD, LCX, RCA), builds tubular volumes of interest (VOI), and produces 5 analysis outputs per patient per vessel.

---

## Overview

| Step | Module | Description |
|---|---|---|
| 1 | `dicom_loader.py` | Load DICOM series → (Z,Y,X) HU volume |
| 2 | `seed_picker.py` | Interactive seed picker (ostium + waypoints) |
| 3 | `centerline.py` | Frangi vesselness + Dijkstra centerline extraction |
| 4 | `pcat_segment.py` | Tubular VOI construction + PCAT HU filtering |
| 5 | `export_raw.py` | Export full-res VOI as `.raw` + metadata JSON |
| 6 | `visualize.py` | 5 analysis outputs (3D, CPR, histogram, radial profile) |
| 7 | `run_pipeline.py` | Orchestrates all steps for single or batch patients |

---

## Requirements

- Python 3.10+
- [Anaconda](https://www.anaconda.com/) base environment (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Quickstart

### Step 1 — Pick Seeds (one-time per patient)

Launch the interactive seed picker to identify the ostium and 1–3 waypoints per vessel:

```bash
python pipeline/seed_picker.py \
    --dicom Rahaf_Patients/1200.2 \
    --output seeds/patient_1200.json
```

**Seed Picker Controls:**

| Key / Action | Effect |
|---|---|
| `1` / `2` / `3` | Switch active vessel: LAD / LCX / RCA |
| Left-click on any view | Add seed point (ostium first, then waypoints) |
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

### Step 2 — Run the Pipeline

**Single patient:**
```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/patient_1200.json \
    --output output/patient_1200 \
    --prefix patient1200
```

**All 3 patients (batch):**
```bash
python pipeline/run_pipeline.py --batch
```

---

## Outputs

Each patient run produces the following in `output/<patient_id>/`:

| File | Description |
|---|---|
| `<prefix>_<VESSEL>_3d_voi.png` | **Output 1** — 3D render of tubular VOI (unfiltered HU) |
| `<prefix>_<VESSEL>_voi.raw` | **Output 2** — Full-res VOI volume (int16, same dims as CCTA) |
| `<prefix>_<VESSEL>_voi_meta.json` | Metadata for the `.raw` file (shape, spacing, dtype) |
| `<prefix>_<VESSEL>_cpr_fai.png` | **Output 3** — Curved Planar Reformat (CPR) with FAI colormap |
| `<prefix>_<VESSEL>_hu_histogram.png` | **Output 4** — HU distribution histogram (FAI range highlighted) |
| `<prefix>_<VESSEL>_radial_profile.png` | **Output 5** — Radial HU profile (1mm rings, 0–20mm from vessel wall) |
| `<prefix>_<VESSEL>_summary.png` | Summary 4-panel figure (all outputs combined) |

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

## Notes

- DICOM data and output files are gitignored — only code and seed templates are tracked.
- 3D renders require `pyvista`: `pip install pyvista`. If not installed, Output 1 is skipped with a warning.
- The pipeline is designed to run on the Anaconda base environment with Python 3.12.
