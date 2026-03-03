# Literature Review: Automated PCAT Segmentation and Coronary Centerline Extraction

**Project**: PCAT Segmentation Pipeline — MolloiLab  
**Date**: February 2026  
**Purpose**: Understand the algorithmic methods behind commercial CCTA PCAT analysis software (ShuKun Technology, Siemens syngo.via) to benchmark our pipeline and identify upgrade paths.

---

## 1. Background: Fat Attenuation Index (FAI) and PCAT

### Biological Basis

Pericoronary adipose tissue (PCAT) surrounds the coronary arteries and communicates bidirectionally with the vessel wall through paracrine signalling. When a coronary artery is inflamed, it inhibits adipogenesis in the immediately adjacent fat, causing a reversible phenotypic shift from the lipid phase toward the aqueous phase. This shift is detectable on CT as an *increase* in HU (less negative) of the perivascular fat.

The **Fat Attenuation Index (FAI)** quantifies this signal as the mean HU of all adipose-range voxels in the pericoronary volume of interest (VOI).

### Foundational Publication

> Oikonomou EK, Marwan M, Desai MY, et al. "Non-invasive detection of coronary inflammation using computed tomography and prediction of residual cardiovascular risk." *Lancet*. 2018;392(10151):929-939. PMID: 30170852

**Key technical specifications validated in this paper (and adopted by our pipeline):**

| Parameter | Value | Notes |
|---|---|---|
| HU range for fat | **−190 to −30 HU** | Adipose tissue window |
| Radial distance | **1× vessel diameter** from outer wall | = mean_radius from centerline outer boundary |
| LAD/LCX segment | **Proximal 40 mm** | From ostium |
| RCA segment | **10–50 mm** (proximal 40 mm, skip first 10 mm) | Avoids aortic wall pulsation artifact |
| Reproducibility | ICC = **0.987** (intraobserver), **0.980** (interobserver) | Excellent |
| Clinical cut-off | **FAI ≤ −70.1 HU** = low-risk; **FAI > −70.1 HU** = high cardiac mortality risk | Validated in CRISP-CT trial |

The FAI cut-off of **−70.1 HU** was validated in the CRISP-CT prospective study (n=1,872) as independently predicting cardiac death (HR 9.04, 95% CI 2.12–38.6, p=0.003).

### Commercialisation

The Oxford methodology has been commercialised by **Caristo Diagnostics** (Oxford, UK). Their FDA-cleared tool (CaRi-Heart) implements the exact methods above.

---

## 2. Commercial Software: ShuKun Technology

### Product

**"Peri-coronary Adipose Tissue Analysis Tool"** — ShuKun Technology Co., Ltd., Beijing, China.  
Companion product: **CoronaryDoc®-FFR** (CT-based fractional flow reserve).

### Published Evidence

Two key studies with explicit ShuKun attribution:
- Huang et al. (2025), PMID 41163958 — *Lesion-specific PCAT radiomics for MACE prediction*
- PMID 39696214 — *PCAT radiomics in stable CAD*

### Technical Methods (inferred from papers)

#### PCAT Definition
- HU range: −190 to −30 (consistent with Oxford standard)
- Radial distance: outer vessel wall + 1 vessel diameter
- **Lesion-specific approach**: PCAT measured around individual stenotic plaques (identified by CT-FFR < 0.8), not just fixed proximal segments

#### Radiomic Feature Extraction
The tool extracts **93 radiomic features** per VOI segmentation:

| Feature Class | Examples |
|---|---|
| First-order statistics | Mean, minimum, maximum, total energy, percentiles |
| GLCM | Contrast, correlation, entropy, homogeneity |
| GLSZM | Small zone emphasis, large zone high grey level |
| GLRLM | Run length non-uniformity, long run emphasis |
| NGTDM | Coarseness, complexity, busyness |
| GLDM | Dependence variance, dependence entropy |

#### Downstream ML Pipeline
1. Feature selection: Pearson correlation filtering (|r| > 0.95) + Lasso regression
2. Normalisation: Min-Max scaling
3. Classification: **XGBoost** with 10-fold cross-validation
4. Outcome: MACE (Major Adverse Cardiovascular Events) prediction

#### Centerline Extraction
ShuKun integrates with CoronaryDoc®-FFR for coronary tree extraction. The exact algorithm is proprietary but, based on the speed of the combined pipeline and the state of the field, almost certainly uses **deep learning** (CNN-based vessel segmentation + automatic ostia detection).

---

## 3. Commercial Software: Siemens syngo.via

### Two Generations Compared

> Weichsel J, et al. "CT-based coronary plaque analysis — comparison of two different software tools." *European Radiology*. 2024. PMID: 38248031

#### Tool #1 — syngo.via Frontier CT Coronary Plaque Analysis (v5.0.2)
- Semi-automated: automatic heart isolation + centerline detection, but requires **manual** correction
- Manual selection of vessel section of interest
- Auto-suggested inner/outer wall contours, then manual refinement
- Processing time: **~459 seconds/case**
- Inter-observer variability: **22.8%**

#### Tool #2 — Successor CT Coronary Plaque Analysis (prototype/next-gen)
- **Fully automated deep learning** for centerline extraction and lumen/vessel wall contouring
- No manual interaction required (correction possible but optional)
- Processing time: **~208 seconds/case** (55% faster than Tool #1)
- Inter-observer variability: **2.3%** (10× more reproducible)

#### Plaque HU Thresholds (for reference)
| Plaque Type | HU Range |
|---|---|
| Calcified | > 350 HU |
| Fibrous | 30–350 HU |
| Lipid / PCAT | < 30 HU |
| PCAT fat range | −190 to −30 HU |

#### Key Insight
The move from Tool #1 → Tool #2 demonstrates the quantified benefit of deep learning for centerline extraction: **55% faster, 10× more reproducible**. The deep learning architecture is proprietary (encoder-decoder type inferred from published patents).

---

## 4. State-of-the-Art Centerline Extraction Algorithms

### 4.1 Rotterdam Benchmark Framework

> Schaap M, Metz CT, van Walsum T, et al. "Standardized evaluation methodology and reference database for evaluating coronary artery centerline extraction algorithms." *Medical Image Analysis*. 2009;13(5):701-714.

The **Rotterdam Coronary Artery Algorithm Evaluation Framework** established standardised metrics and benchmarked 13 algorithms on 32 cardiac CTA datasets. It is the canonical comparison framework for coronary centerline methods.

**Evaluation metrics:**
- **OV** (Overlap): fraction of centerline within 1 voxel of ground truth
- **OF** (Overlap at Forking points): OV at bifurcations
- **OT** (Overlap at Tips): OV at vessel tips
- **AI** (Average Interslice distance): positional accuracy in mm

### 4.2 Classical Fast Methods

#### Minimal Path / Fast Marching
Cohen & Kimmel (1997) + Deschamps & Cohen (2001). Finds globally optimal path between two endpoints by propagating a wavefront through a cost field derived from vesselness. Faster than Dijkstra on dense graphs because it processes each voxel at most once (O(n log n) with heap vs O(n²) naive Dijkstra).

**scikit-fmm** is the Python implementation used in our pipeline upgrade.

#### Optimally Oriented Flux (OOF) Filter
> Law MWK, Chung ACS. "Three dimensional curvilinear structure detection using optimally oriented flux." *ECCV*. 2008.

OOF computes flux through a spherical surface rather than Hessian eigenvalues (Frangi). Advantages:
- **2–3× faster** than Frangi for equivalent accuracy
- Better vessel boundary detection for varying diameters
- O(n) per scale vs O(n·σ) for Frangi
- No Python package with maintained coronary-specific implementation exists as of 2026

> Jerman T, Pernuš F, Likar B. "Beyond Frangi: an improved multiscale vesselness filter." *Medical Imaging 2015*, SPIE Proceedings.

Jerman et al. showed OOF significantly outperforms Frangi in vessel segmentation, with better response uniformity and edge detection.

### 4.3 Deep Learning Methods (2024–2025)

#### Deep Reinforcement Learning (Zhang et al. 2025)
> PMID 39888471

- Actor-Critic architecture; continuous action space
- Overcomes artifacts and calcified plaques better than discrete-action methods
- **OV = 95.7%, OF = 93.6%, OT = 97.3%, AI = 0.22 mm**
- Speed: seconds per case with GPU

#### Lightweight Dual-CNN (Liu et al. 2025)
> Proc SPIE 13407

- Two lightweight 3D CNNs in parallel:
  1. Direction predictor (forward/backward orientation)
  2. Vessel distance map generator (stopping criterion)
- Achieves ~95% sensitivity and overlap
- Speed: **8–15 seconds/case**
- Requires pre-trained weights (training data: proprietary)

#### BEA-CACE (Branch-Endpoint-Aware, Zhang et al. 2025)
> PMID 40751109

- Double-DQN (deep Q-network) tracker + 3D dilated CNN detector
- Simultaneously extracts centerline and estimates vessel radius
- End-to-end trainable

#### CenterlineNet (Rjiba et al. 2020)
- Patch-based CNN; detects main + side branches without explicit segmentation
- Works directly on CTA volumes

### 4.4 Speed Comparison

| Method | Speed | Hardware | Accuracy |
|---|---|---|---|
| ~~Frangi + Dijkstra (full volume)~~ | ~~10+ min~~ | CPU | — (replaced) |
| ~~Frangi + Dijkstra (ROI-cropped)~~ | ~~10–30s~~ | CPU M3 | — (replaced) |
| **Frangi + Fast Marching (current pipeline, ROI-cropped)** | **~10–30s** | **CPU M3** | **High** |
| OOF + Fast Marching | ~3–8s | CPU | High |
| Lightweight CNN (Liu 2025) | 8–15s | GPU | ~95% |
| Deep RL (Zhang 2025) | seconds | GPU | 95.7% OV |
| Siemens syngo.via Tool #2 | ~208s/case | Clinical workstation | Low variability |

---

## 5. Implications for Our Pipeline

### What We Match ✅
- HU range: −190 to −30 HU (Oxford/Caristo standard)
- Radial distance: 1× vessel diameter from outer wall
- Proximal segments: 40 mm LAD/LCX, 10–50 mm RCA
- Clinical cut-off: −70.1 HU reported per vessel

### Where We Differ from Commercial Tools
| Gap | Commercial Solution | Our Current Approach | Upgrade Path |
|---|---|---|---|
| Seed picking | Automatic (DL ostia detection) | TotalSegmentator (auto) + optional manual review | CNN landmark detection (future) |
| Centerline algo | Fast Marching or DL | ✅ **Fast Marching** (scikit-fmm, ROI-cropped ~10–30 s) | Deep RL / CNN (future) |
| Vessel enhancement | OOF or DL | Frangi (ROI-cropped) | OOF (needs C++ binding) |
| Radiomic features | 93 features (XGBoost) | Mean HU, std, percentiles | pyradiomics integration (future) |
| Processing speed | ~208s/case (Siemens) | ~30–60s/case | Within range of commercial tools |
### Recommended Future Upgrades (Priority Order)
1. **Add pyradiomics** for 93-feature radiomic extraction per VOI
2. **Automatic ostia detection** — CNN landmark or atlas-based (3D Slicer VMTK)
3. **OOF filter** — implement via `morphsnakes` approximation or C++ binding
4. **Deep learning centerline** — replace Frangi+FMM with trained vessel tracker for calcified/artifact cases

---

## 6. Key References

1. Oikonomou EK et al. *Lancet* 2018. PMID: 30170852 — FAI methodology, −70.1 HU cut-off
2. Huang et al. *[Journal]* 2025. PMID: 41163958 — ShuKun PCAT radiomics for MACE
3. Weichsel J et al. *Eur Radiol* 2024. PMID: 38248031 — Siemens syngo.via comparison
4. Schaap M et al. *Med Image Anal* 2009;13:701-714 — Rotterdam benchmark framework
5. Law MWK, Chung ACS. *ECCV* 2008 — OOF filter
6. Jerman T et al. *SPIE Med Imaging* 2015 — OOF vs Frangi comparison
7. Zhang et al. *[Journal]* 2025. PMID: 39888471 — Deep RL centerline (Actor-Critic)
8. Liu CC et al. *Proc SPIE* 13407. 2025 — Lightweight CNN centerline (8–15s)
9. Zhang et al. *[Journal]* 2025. PMID: 40751109 — BEA-CACE double-DQN

---

## 7. PCAT as a Cardiovascular Biomarker: Study Context

### 7.1 Epidemiological Evidence

Pericoronary adipose tissue volume and attenuation are independently associated with coronary artery disease severity. Multiple large cohort studies establish the clinical context:

**CRISP-CT (Oikonomou et al. 2018, *Lancet*, PMID 30170852)**  
The foundational prospective study. In 1,872 patients undergoing CCTA for suspected CAD, FAI around the right coronary artery (RCA-FAI) independently predicted cardiac death over 5 years (HR 9.04, 95% CI 2.12–38.6). This established FAI as the first non-invasive imaging biomarker of coronary inflammation detectable before atherosclerotic plaque formation.

**CRISP-CT mechanistic validation:**  
The biological mechanism was confirmed in matched histological specimens: perivascular fat biopsied adjacent to inflamed coronary segments showed significantly reduced lipid droplet size, reduced expression of adipogenic transcription factors (PPARgamma, FABP4), and increased expression of inflammatory cytokines (IL-6, TNF-alpha) — directly corresponding to the HU elevation measured on CT.

**ORFAN Trial (Oikonomou et al. 2023, *Nature Cardiovascular Research*)**  
The first prospective trial to test AI-enhanced PCAT analysis (CaRi-Heart score, Caristo Diagnostics). The trial demonstrated that coronary inflammation detected by FAI was a stronger predictor of MACE than conventional risk scores (ASCVD PCE score, SRS). The AI score added incremental predictive value beyond calcium scoring.

### 7.2 Material Decomposition and Spectral CT for PCAT

Standard single-energy CT measures a single integrated HU value that conflates contributions from water, lipid, protein, and calcium. **Spectral CT** (dual-energy or photon-counting detector CT) enables material decomposition to separate these components, potentially providing more specific markers of inflammation:

**Water-lipid decomposition:**  
Fat tissue is predominantly triglycerides (~86% lipid by weight). Inflamed PCAT shifts from lipid-dominant to more aqueous composition. On spectral CT, the **lipid map** (lipid density image) would directly quantify this shift — a fundamentally more specific measurement than the integrated FAI HU.

**Photon-counting detector CT (PCD-CT):**  
Next-generation scanners (Siemens NAEOTOM Alpha, GE HealthCare Revolution CT) provide simultaneous multi-energy data at full resolution. PCD-CT generates:
- Virtual monoenergetic images (VMI) at any keV
- Material decomposition maps (water, iodine, lipid, calcium)
- Effective atomic number (Z-eff) maps

For PCAT: VMI at 70 keV matches conventional CT noise/contrast characteristics while material maps add specificity. Our pipeline's current data (Siemens syngo.via, 'mono 70 keV' series label) is consistent with virtual monoenergetic reconstruction from a dual-energy or spectral acquisition.

**Key implication for our pipeline:**  
The -190 to -30 HU fat window is defined for conventional 120 kVp polychromatic CT. On VMI, the same tissue will appear slightly different due to the energy-dependent HU. The threshold needs validation at 70 keV VMI (expected shift: approximately +5 to +15 HU relative to 120 kVp).

### 7.3 PCAT Volume vs. Attenuation

Two separate PCAT phenotypes are measured:

| Measure | Biological Meaning | Clinical Association |
|---|---|---|
| **PCAT attenuation (FAI, HU)** | Inflammatory phenotypic shift in adipocytes | Acute inflammation, plaque vulnerability, future MACE |
| **PCAT volume (cm3)** | Total adipose depot size | Obesity, metabolic syndrome, chronic risk |

These are partially independent: a patient can have high PCAT volume (large depot, obese) but low FAI (non-inflamed), or low volume but high FAI (lean but actively inflamed). Both should ideally be measured. Our pipeline currently computes **attenuation (FAI)** per vessel; volume is derivable from the same VOI mask (voxel count x voxel volume in cm3) and is already present in `compute_pcat_stats` output.

---

## 8. Coronary Artery Inflammation: Field Context

### 8.1 The Vascular Inflammation Hypothesis

The paradigm shift in cardiovascular medicine is the recognition that **atherosclerosis is fundamentally an inflammatory disease**, not merely a lipid storage disorder:

- **Ross R. *NEJM* 1999**: Established the 'response-to-injury' hypothesis — endothelial activation by LDL, oxidative stress, and shear forces initiates an inflammatory cascade
- **Ridker PM et al. *NEJM* 2017 (CANTOS trial)**: Showed anti-inflammatory therapy (canakinumab, IL-1beta antibody) reduced MACE by 15% independent of LDL, proving the causal role of inflammation
- **Libby et al. *Nature* 2021**: Comprehensive review establishing the 'inflammasome' pathway (NLRP3) as the central mediator of plaque vulnerability

### 8.2 Perivascular Adipose Tissue as Paracrine Signalling Hub

**Vasocrine signalling (arterial wall -> fat):**
- Coronary arterial smooth muscle and adventitia release pro-inflammatory mediators (IL-6, TNF-alpha, CXCL10) during atherosclerotic activity
- These diffuse outward into perivascular fat and suppress adipocyte differentiation (inhibit PPARgamma, C/EBPalpha)
- Result: adipocytes become smaller, less lipid-filled -> HU increases toward less-negative values

**Paracrine signalling (fat -> vessel wall):**
- In obese/metabolic syndrome states, PVAT shifts from secreting vasodilatory (adiponectin, NO) to pro-inflammatory (IL-6, TNF-alpha, FABP4) mediators
- This creates a bidirectional amplification loop that accelerates plaque development

### 8.3 PCAT vs. Epicardial Adipose Tissue

| Feature | Pericoronary AT (PCAT) | Epicardial AT (EAT) |
|---|---|---|
| Location | Immediately adjacent to vessel wall | Entire fat depot within pericardium |
| Measurement | HU attenuation in fixed VOI per vessel | Total volume (cm3) within pericardial sac |
| Inflammation signal | Direct (per-vessel FAI) | Indirect (whole depot mean HU) |
| Clinical tool | CaRi-Heart (Caristo), ShuKun | EAT volume tools |

### 8.4 Limitations and Open Questions

1. **Partial volume effects**: Sub-mm coronary vessels are near the spatial resolution limit of CT. Small vessels (LCX, LAD diagonals) have worse SNR.
2. **Cardiac motion artifact**: Motion-blurred voxels partially outside fat range may bias FAI.
3. **Threshold generalisability**: The -70.1 HU FAI cut-off was established on conventional 120 kVp CT; validation on spectral VMI data is an active research area.
4. **LCX underestimation**: The LCX runs adjacent to the left atrial wall; VOI frequently overlaps non-adipose tissue (consistent with our low LCX voxel count of 639 vs. 13,784 for LAD).

---

## 9. Simulation Study Context

### 9.1 Purpose of Our Pipeline

The immediate application context is a **simulation study** characterising PCAT quantification accuracy on CCTA images acquired from a phantom or computational model. The pipeline serves as the measurement tool: given a known ground-truth fat distribution, how accurately does the FAI extraction reproduce the known HU values?

Key validation questions:
- Does the VOI construction correctly capture the pericoronary fat shell?
- How does spatial resolution affect FAI accuracy at the coronary scale (vessel diameter ~3 mm)?
- What is the sensitivity to centerline positioning error (e.g., 1 mm offset)?

### 9.2 Material Decomposition Connection

If the phantom is scanned on a spectral or dual-energy CT, the simulation study could directly test **water-lipid decomposition** as a PCAT measurement method — comparing conventional FAI (integrated HU) to spectral lipid map HU against known ground-truth composition. This would be the first study directly comparing the two methods for PCAT quantification.

---

## 10. Summary Table: Key Papers by Topic

| Topic | Paper | Year | Key Contribution |
|---|---|---|---|
| FAI foundational | Oikonomou et al., *Lancet* | 2018 | FAI definition, -70.1 HU cut-off, CRISP-CT validation |
| FAI AI score | Oikonomou et al., *Nat CV Res* | 2023 | CaRi-Heart AI score, ORFAN trial |
| PCAT radiomics | Huang et al., PMID 41163958 | 2025 | 93-feature radiomic MACE prediction (ShuKun) |
| Siemens syngo.via | Weichsel et al., *Eur Radiol* | 2024 | Tool comparison: DL reduces variability 10x |
| Inflammation trial | Ridker et al., *NEJM* (CANTOS) | 2017 | Causal role of IL-1beta in MACE |
| Coronary centerline | Schaap et al., *Med Image Anal* | 2009 | Rotterdam benchmark framework |
| DL centerline | Zhang et al., PMID 39888471 | 2025 | Deep RL: OV=95.7%, speed=seconds |
| Spectral CT review | Eveson et al., *Br J Radiol* | 2026 | AI integration + quantitative PCAT review |
| PCD-CT plaque | Engel et al., *J Clin Med* | 2026 | FAI >= -70.1 HU and plaque composition |
