# PCAT — Pericoronary Adipose Tissue Pipeline

Automatically measures fat around coronary arteries from a cardiac CT scan (CCTA). Outputs CPR images, HU histograms, radial profiles, and a per-vessel FAI summary.

---

## What does it do?

1. Finds your coronary arteries (LAD, LCX, RCA) automatically
2. Segments the fat tissue around them
3. Measures the **Fat Attenuation Index (FAI)** — a clinical marker for heart inflammation
4. Saves images and statistics you can review and report

> **FAI risk threshold: −70.1 HU.** Values above this (less negative) = higher cardiovascular inflammation risk (Oikonomou et al., Lancet 2018).

---

## Setup (one time)

```bash
git clone https://github.com/MolloiLab/PCAT.git
cd PCAT
pip install -r requirements.txt
```

Get a free TotalSegmentator research licence at:
`https://backend.totalsegmentator.com/license-academic/`

---

## Semi-Automatic Mode (recommended)

This is the standard clinical workflow. The pipeline runs automatically, then pauses so you can review each vessel before the results are saved.

**Step 1 — Run the pipeline**

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --auto-seeds
```

That's it. The pipeline will:
- Detect coronary arteries with TotalSegmentator (~30–60 s)
- Extract centerlines and segment pericoronary fat
- **Pause and open the VOI Editor** — review the auto-segmented fat region for each vessel, then press `s` to approve or `q` to skip
- **Pause and open the CPR Browser** — inspect the cross-sections along the full vessel length, then press `q` to continue
- Save all images and statistics to `output/patient_1200/`

**Step 2 — Check the results**

Open `output/patient_1200/` — you'll find:
- `patient1200_LAD_cpr_fai.png` — CPR with fat overlay (yellow = healthy fat, red = inflamed)
- `patient1200_summary.png` — bar chart of FAI, fat fraction, voxel count for all vessels
- `patient1200_results.json` — numerical FAI statistics

---

## Fully Automatic Mode (batch / no interaction)

Use this for bulk processing or headless servers where no GUI is available.

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

## If TotalSegmentator misses a vessel

Run the seed reviewer to check and fix the auto-detected coronary locations:

```bash
python pipeline/seed_reviewer.py \
    --dicom Rahaf_Patients/1200.2 \
    --seeds seeds/1200_2_auto.json \
    --output seeds/patient_1200_reviewed.json
```

Then pass the corrected seeds to the pipeline:

```bash
python pipeline/run_pipeline.py \
    --dicom Rahaf_Patients/1200.2 \
    --output output/patient_1200 \
    --prefix patient1200 \
    --seeds seeds/patient_1200_reviewed.json
```

**Seed reviewer controls:** `1`/`2`/`3` switch vessel · click moves/adds a seed · `d` deletes a waypoint · `s` saves

---

## VOI Editor controls (opens automatically)

| Key | Action |
|-----|--------|
| `1` | Paint — add voxels to the fat region |
| `2` | Erase — remove voxels |
| `3` | Fat-only paint — only adds fat (−190 to −30 HU) |
| `[` / `]` | Decrease / increase brush size |
| `u` | Undo |
| `s` | Save and continue |
| `q` | Skip without saving |
| Scroll wheel | Change slice |

---

## CPR Browser controls (opens automatically)

| Key / Action | Effect |
|---|---|
| Click on CPR panel | Jump to that vessel position |
| `←` / `→` | Step one point along the vessel |
| `s` | Save a PNG snapshot |
| `q` | Close and continue |

---

## Output files

All outputs land in the directory you set with `--output`:

| File | What it shows |
|------|--------------|
| `*_cpr_fai.png` | CPR with FAI overlay — yellow/red = fat |
| `*_cpr_gray.png` | Grayscale CPR at 6 rotation angles |
| `*_cpr_native_rot*.png` | Curved CPR (Syngo.via style) |
| `*_hu_histogram.png` | HU distribution of fat voxels |
| `*_radial_profile.png` | Fat HU vs. distance from vessel wall |
| `*_summary.png` | Bar charts: FAI, fat fraction, voxel count |
| `*_results.json` | Numerical FAI statistics |

---

## Reading CPR images

- **Vessel runs top → bottom.** Top = ostium (origin), bottom = distal end.
- **Yellow/red regions** = pericoronary fat (FAI range −190 to −30 HU). Yellow = more negative HU (healthier). Red = less negative (more inflamed).
- **White dashed vertical line** = vessel centerline axis.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "too few centerline points" | Re-run `seed_reviewer.py` — seed may be off-vessel |
| NaN mean HU in results | No fat voxels in VOI; vessel may be heavily calcified |
| "only N vessels found" warning | Run `seed_reviewer.py` and add the missing vessel manually |
| 3D render skipped | `pip install pyvista` |
| VOI editor / CPR browser won't open | Add `--skip-editor --skip-cpr-browser` (headless server) |
| TotalSegmentator fails | Place seeds manually: `python pipeline/seed_picker.py --dicom ... --output seeds/patient.json` |

---

## Patient data included

| Patient | DICOM folder | Slices |
|---------|-------------|--------|
| 1200 | `Rahaf_Patients/1200.2/` | 405 |
| 2 | `Rahaf_Patients/2.1/` | 149 |
| 317 | `Rahaf_Patients/317.6/` | 399 |

To add a new patient, just change `--dicom` to the new DICOM folder and `--output` to a new folder.

---

## Adding multiple patients (batch)

Edit `PATIENT_CONFIGS` in `pipeline/run_pipeline.py` (~line 88):

```python
PATIENT_CONFIGS = [
    {"patient_id": "1200", "dicom": "Rahaf_Patients/1200.2", "seeds": "seeds/patient_1200.json", "output": "output/patient_1200", "prefix": "patient1200"},
    {"patient_id": "999",  "dicom": "New_Patients/patient999", "seeds": "seeds/patient_999.json", "output": "output/patient_999", "prefix": "patient999"},
]
```

Then run:

```bash
python pipeline/run_pipeline.py --batch --auto-seeds
```
