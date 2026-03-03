# Coronary Artery Inflammation and Pericoronary Adipose Tissue: Field Review

**Project**: PCAT Segmentation Pipeline — MolloiLab  
**Author**: Shu Nie  
**Date**: March 2026  
**Scope**: In-depth review of the PCAT/FAI field — clinical motivations, biological basis, landmark trials, prognostic evidence, research groups, study design patterns, patient selection, technical limitations, emerging applications, PCAT radiomics, photon-counting CT, commercial landscape, and open questions. Synthesised from ~85 papers in the coronary inflammation collection, online literature search (2023–2026), and cross-referencing of major reviews.

---

## Part I: Clinical Motivation — Why Measure Coronary Inflammation?

### 1.1 Residual Inflammatory Risk

Despite optimal lipid-lowering therapy with statins, a substantial **residual cardiovascular risk** persists. The CANTOS trial proved that this residual risk is driven by **inflammation independent of LDL cholesterol**: canakinumab (IL-1β antibody) reduced MACE by 15% without affecting LDL levels. This established inflammation as a direct, causal therapeutic target.

The colchicine trials subsequently confirmed a practical, affordable anti-inflammatory pathway:
- **COLCOT**: −23% MACE post-MI
- **LoDoCo2**: −31% MACE in chronic CAD

The clinical need is therefore to **identify patients with active coronary inflammation** who would benefit from anti-inflammatory therapy — but no widely available non-invasive test existed to localise inflammation to specific coronary arteries until FAI.

### 1.2 Why PCAT Is Uniquely Informative

Pericoronary adipose tissue provides a **vessel-specific, non-invasive readout of coronary inflammation** that systemic biomarkers cannot:

| Biomarker | Signal Source | Limitation |
|---|---|---|
| hsCRP (serum) | Systemic inflammation | Non-specific — elevated in infection, obesity, autoimmune disease |
| IL-6 (serum) | Systemic cytokine burden | Cannot localise to specific coronary artery |
| 18F-FDG PET | Metabolically active inflammation | Poor spatial resolution, cardiac/respiratory motion, myocardial uptake confounds |
| **FAI (CCTA)** | **Per-vessel local coronary inflammation** | Specific to individual coronary artery segment; spatial resolution to single vessel |

FAI is the only clinical measurement that localises inflammation to a **specific coronary artery segment** non-invasively. A stenotic LAD with high FAI is a fundamentally different risk entity than the same stenosis with low FAI.

### 1.3 The Treatment Pathway FAI Enables

```
CCTA acquired → FAI computed per-vessel → FAI > −70.1 HU detected
       ↓
Classification: Active coronary inflammation
       ↓
Clinical action: Consider colchicine / statin intensification / earlier follow-up
```

This pathway is deployed clinically via CaRi-Heart (Caristo Diagnostics, Oxford, UK) — the FDA-cleared commercial implementation of the Oxford FAI methodology, with CPT codes assigned in 2025 and Medicare reimbursement finalized for 2026.

---

## Part II: Biology of PCAT — What FAI Measures

### 2.1 PCAT Anatomy

**Pericoronary adipose tissue (PCAT)** is the adipose tissue immediately surrounding the coronary arteries, between the coronary vessels and the epicardial surface. It lacks a fascial barrier between the fat and the adventitia, creating a direct exchange pathway for molecular signaling. PCAT is vascularized by the vasa vasorum, sharing the same microvascular supply as the adventitia.

| Feature | PCAT | Epicardial Adipose Tissue (EAT) | Paracardial Fat |
|---|---|---|---|
| **Location** | ~3–5 mm shell around specific vessel segment | Entire pericardial sac | Outside the pericardium |
| **Measurement** | Mean HU (attenuation = FAI) | Total volume (cm³) | Volume, rarely HU |
| **Clinical signal** | Per-vessel acute inflammatory state | Whole-heart chronic metabolic risk | Limited clinical relevance |
| **Segmentation** | Vessel-specific VOI (centerline-based) | Pericardial sac segmentation | Not routinely measured |

### 2.2 Bidirectional Vessel-Fat Signalling

#### Vasocrine Signalling: Vessel Wall → Fat

When the coronary artery is inflamed:
1. Adventitial macrophages and smooth muscle cells secrete **IL-6, TNF-α, CXCL10, VEGF**
2. These mediators diffuse outward into adjacent PCAT
3. IL-6 and TNF-α suppress adipocyte differentiation:
   - Inhibit **PPARγ** (master transcription factor for fat cell maturation)
   - Inhibit **C/EBPα** (co-factor for adipogenesis)
   - Inhibit **FABP4** (fatty acid binding protein, marker of mature adipocyte)
4. Adipocytes remain immature, smaller, with less stored lipid
5. **Result on CT**: fat voxels shift from lipid-dominant (HU ≈ −90 to −70) toward more aqueous composition (HU ≈ −60 to −40) — this is the FAI signal

The FAI increase reflects a **genuine molecular phenotypic shift** in the fat cells, validated histologically by Antonopoulos et al. (2017): perivascular fat sampled adjacent to inflamed coronary segments showed reduced lipid droplet size (p<0.001), reduced PPARγ (−2.3-fold), reduced FABP4 (−1.8-fold), and increased IL-6 (+3.1-fold) and TNF-α (+2.7-fold).

#### Paracrine Signalling: Fat → Vessel Wall

In obesity and metabolic syndrome, the PVAT itself becomes dysfunctional:

| Normal PVAT | Dysfunctional PVAT (obese/inflamed) |
|---|---|
| Secretes **adiponectin** (anti-inflammatory, vasodilatory) | Adiponectin ↓ |
| Secretes **NO** (endothelial relaxation) | NO ↓ |
| Secretes **omentin** (insulin-sensitising) | Leptin ↑ |
| Low macrophage infiltration (M2 phenotype) | High macrophage infiltration (M1 phenotype) |
| — | IL-6, TNF-α, FABP4 ↑ |
| — | Reactive oxygen species ↑ |

This creates a **bidirectional amplification loop**: inflamed vessels trigger PCAT dysfunction, which in turn secretes pro-inflammatory mediators that further accelerate plaque development.

### 2.3 Spatial Heterogeneity

#### Proximal Segments

The proximal coronary segments (LAD 0–40 mm, LCX 0–40 mm, RCA 10–50 mm) are the clinical focus because:
1. **Most plaques form here**: hemodynamic shear stress at bifurcations and proximal curves
2. **Clinical consequence**: proximal stenoses cause more downstream ischemia
3. **CT resolution**: proximal vessels are larger (3–5 mm diameter) → better SNR for FAI

#### RCA vs. LAD vs. LCX: Different Biology

The clinical literature focuses primarily on RCA-FAI because:
- RCA has the most pericoronary fat (largest fat depot, cleaner VOI)
- LAD runs in the anterior interventricular groove — also well-studied
- LCX runs in the atrioventricular groove adjacent to the left atrial wall → VOI contamination more common

Reference values (Ma et al. 2020): LAD −92.4 HU, LCX −88.4 HU, RCA −90.2 HU. FAI increases linearly with tube voltage.

#### Lesion-Specific vs Fixed Proximal VOI

Rather than fixed proximal segments, measuring PCAT adjacent to each individual plaque provides:
- **Higher specificity**: PCAT around a stable fibrous plaque may be low even in the same vessel as an unstable plaque with high PCAT
- **Better MACE prediction**: the PCAT signal is strongest immediately adjacent to the vulnerable plaque

Li et al. (2025) compared PCATMA (Pericoronary Adipose Tissue Mean Attenuation — not constrained by fat threshold) with FAI and found PCATMA showed significant differences for non-calcified plaque (P<0.001) and mixed plaque (P=0.047), while FAI did not — suggesting threshold-free measurement may be more sensitive.

### 2.4 The Inflammasome Pathway

The **NLRP3 inflammasome** is the central molecular sensor in atherosclerotic plaques:

```
Cholesterol crystals / oxidised LDL
       ↓
 NLRP3 inflammasome activation
       ↓
 Caspase-1 activation
       ↓
 IL-1β + IL-18 maturation & secretion
       ↓
 Downstream: IL-6, CRP, MMP secretion
       ↓
 Fibrous cap thinning → plaque vulnerability
```

NLRP3 is detectable in perivascular fat macrophages adjacent to vulnerable plaques — directly linking the fat compartment to the vessel wall inflammatory milieu that FAI measures.

---

## Part III: Clinical Evidence — FAI Validation and Prognostic Value

### 3.1 Foundational Studies

#### Antonopoulos 2017 (Histological Validation)

> Antonopoulos et al. *Sci Transl Med* 2017. n=453 cardiac surgery patients.

The landmark study that introduced FAI, providing histological validation that PCAT attenuation changes reflect genuine molecular phenotypic shifts in adipocytes: reduced lipid droplet size (p<0.001), reduced PPARγ (−2.3-fold), reduced FABP4 (−1.8-fold), and increased IL-6 (+3.1-fold) and TNF-α (+2.7-fold) in fat adjacent to inflamed coronary segments.

#### CRISP-CT 2018 (FAI Definition and Prognostic Validation)

> Oikonomou et al. *Lancet* 2018. n=1,872 (Erlangen derivation + Cleveland Clinic validation).

- **RCA-FAI** independently predicted cardiac death at 5-year follow-up: **HR 9.04** (95% CI 2.12–38.6, p=0.003)
- FAI added incremental prognostic value beyond CACS, Gensini score, and Framingham Risk Score
- **FAI cut-off: −70.1 HU** identified by ROC analysis (AUC = 0.76 for cardiac death)
- Reproducibility: ICC = 0.987 intraobserver, 0.980 interobserver
- Defined the exact technical parameters (−190 to −30 HU fat window, proximal 40 mm VOI, 1× vessel diameter radial extent)

#### ORFAN 2023 & ORFAN Extended 2024 (AI-Enhanced FAI)

> Oikonomou et al. *Nature Cardiovascular Research* 2023. n=3,324.
> Chan et al. *Lancet* 2024. n=40,091.

The largest PCAT/FAI study to date — n=**40,091** consecutive CCTA patients from **8 UK NHS hospitals**, with two nested cohorts:

- **Cohort A** (n=40,091): Median follow-up **2.7 years**. 81.1% had **no obstructive CAD**, yet this group accounted for **66.3% of all MACE** and **63.7% of cardiac deaths** — proving that inflammatory risk extends far beyond stenosis-based risk.
- **Cohort B** (n=3,393): Median follow-up **7.7 years**. Validated FAI Score in all 3 coronary territories (LAD, LCX, RCA).

Key findings from the full cohort:

| Finding | Metric |
|---|---|
| FAI Score in ANY artery predicted cardiac mortality/MACE | Independent of risk factors AND extent of CAD |
| 3 inflamed arteries vs none — cardiac mortality | **HR 29.8** |
| 3 inflamed arteries vs none — MACE | **HR 12.6** |
| AI-Risk classification (FAI Score + plaque burden + risk factors) | NRI **0.38** for cardiac mortality |
| AI-Risk very high vs low/medium — cardiac mortality | **HR 6.75** |
| AI-Risk very high vs low/medium — MACE | **HR 4.68** |

Used **CaRi-Heart v2.5** (Caristo Diagnostics) for FAI Score computation — not raw FAI but a standardised score adjusted for technical/anatomical/demographic factors.

**Key conclusion**: FAI Score captures inflammatory risk **beyond** current risk stratification, especially in patients **without** obstructive CAD — the very patients missed by traditional stenosis-based evaluation.

#### Sagris 2022 Meta-Analysis

> Sagris et al. 2022. 20 studies, n=7,797.

FAI significantly higher around unstable vs stable plaques, confirming the FAI signal is robust and reproducible across diverse study populations and scanner platforms.

### 3.2 Prognostic Applications

#### Risk Stratification & Reclassification

**Coerkamp et al. (2025)**: FAI reclassified **62% of patients** — 22% upgraded to higher risk, 40% downgraded. Used CaRi-Heart commercially. Demonstrates that FAI meaningfully changes clinical decisions in a majority of patients.

#### Culprit Lesion Identification (ACS)

- **Li et al. (2025)**: FAI identified culprit lesions in ACS with optimal cutoff of −77 HU. AUC 0.970 when combined with stenosis severity (vs 0.939 for stenosis alone). Non-calcified and mixed plaques showed higher FAI.
- **Yang et al. (2025)**: FAIlesion significantly higher at culprit vs non-culprit sites.
- **Huang et al. (2023)**: Combining FAI with CT-FFR significantly improved culprit lesion identification beyond anatomical assessment alone.

#### Plaque Vulnerability

- **Luo et al. (2026)**: FAI predicts vulnerable plaque characteristics and adverse outcomes.
- **PCAT+OCT combined (2025)**: PCAT attenuation associated with plaque vulnerability using combined CCTA and OCT imaging.
- **Wang et al. (2025)**: Significant correlations between plaque vulnerability features and multiparametric PCAT indices.
- **Li et al. (2025)**: FAI correlates with changes in plaque components before and after coronary plaque formation, interacting with both volume and composition of newly formed plaques, particularly the necrotic core.

#### Special Populations

**Diabetes**:
- **Zhang et al. (2025)**: Longer diabetes duration independently associated with increased FAI in all three major coronary arteries. Linear dose-response relationship.
- **Feng et al. (2026)**: FAI mediated the relationship between TyG index and OSA risk in type 2 diabetes.
- **Wang et al. (2026)**: FAI provided incremental prognostic value in diabetic patients with non-obstructive CAD.
- **Lesion-specific FAI in T2DM (2024)**: FAI for MACE prediction in type 2 diabetes.

**CKD**:
- **Lu et al. (2025)**: High pFAI independently predicted MACE (HR 1.65–2.72) and cardiovascular mortality (HR 2.12–2.90) across all 3 vessels in CAD + CKD patients. Used ShuKun technology. Median follow-up 4.57 years.

**HFpEF**:
- **Yuasa et al. (2025)**: FAI predicted hospitalization for **HFpEF** (heart failure with preserved ejection fraction) — a novel application beyond traditional atherosclerotic endpoints.

**Young patients**:
- **2025 (BMC Cardiovasc Disord)**: FAI for MACE prediction in young people — extending the applicability beyond the typical >50 year age group.

#### Non-Obstructive CAD / MINOCA

> Diau & Lange (2025): Comprehensive review of coronary inflammation in non-obstructive CAD and MINOCA.

Key findings across the literature:
- The ORFAN 40,091-patient study showed more cardiac deaths in the non-obstructive group (81% of the cohort)
- EAT, PCAT, FAI, and AI-Risk algorithms all provide value in this population
- **Tognola et al. (2025)**: PCAT and EAT contribute to coronary inflammation in MINOCA patients — CCTA can detect localised inflammation through attenuation changes
- **Port et al. (2025)**: First reported case of abnormal FAI in hypereosinophilic syndrome causing INOCA

### 3.3 FAI Beyond Atherosclerosis

#### Vasospasm

The **Shimokawa group (Tohoku University)** pioneered this application:
- **Ohyama et al. (2016–2017)**: Used 18F-FDG PET/CT to show that coronary perivascular FDG uptake (indicating inflammation) was significantly increased at spastic coronary segments. Coronary PCAT volume was increased at spastic segments in vasospastic angina (VSA) patients, with significant positive correlation between PCAT volume and vasoconstricting response to acetylcholine.
- After 23 months of medical treatment, coronary perivascular FDG uptake decreased — demonstrating FAI dynamics with treatment.
- Established that **local adventitial inflammation**, not systemic inflammation, drives coronary spasm.

#### Myocarditis and Pericarditis

- **Baritussio et al. (2021)**: Patients with clinically suspected myocarditis presenting with infarct-like symptoms had significantly elevated FAI values compared to controls. One of the first applications in non-ischaemic inflammatory heart conditions.

#### Cardiac Transplant

- **Moser et al. (2023)**: Perivascular fat attenuation independently predicted cardiac mortality and need for re-transplantation in heart transplant recipients.
- **Lassandro et al. (2026)**: Dynamic FAI progression predicted CAV development and outcomes — suggesting FAI could be valuable for non-invasive surveillance of transplant recipients.

#### COVID-19 and Atrial Fibrillation

- Multiple studies (2020–2022) examined coronary inflammation post-COVID-19 infection, finding elevated FAI values that correlated with cardiovascular complications.
- Emerging research explores connections between left atrial PCAT and AF substrate, though this remains early-stage.

### 3.4 Serial Measurements & Treatment Monitoring

- **Yoshihara et al. (2025)**: Serial FAI measurements in spontaneous coronary artery dissection showed persistent elevation correlating with disease progression.
- **Shimokawa group**: Demonstrated decreased perivascular inflammation (FDG uptake, correlated with FAI) after 23 months of medical treatment for vasospastic angina.
- **LoDoCo2 CT substudy (Fiolet et al. 2025)**: Despite colchicine reducing MACE by 31% in the main LoDoCo2 trial, PCAT attenuation **failed to detect** its anti-inflammatory effect. If FAI cannot capture the signal of a proven anti-inflammatory drug, its utility as a **treatment monitoring biomarker** is questionable. This directly supports the argument that HU-based FAI has fundamental sensitivity limitations.

---

## Part IV: Limitations — Why FAI Is Not Enough

### 4.1 Technical Confounders

FAI measurements are significantly affected by technical parameters, with quantified magnitudes from specific studies:

#### Tube Voltage/kVp
- **Nie & Molloi (2025)**: HU variance of **21.9%** across 80–135 kV for identical tissue composition
- **Ma et al. (2020)**: FAI increases linearly with tube voltage (less negative at higher kV)
- **Etter et al. (2022)**: Required conversion factors relative to 120 kVp: 1.267 (80 kVp), 1.08 (100 kVp), 0.947 (140 kVp)

#### Reconstruction Kernel and Algorithm
- **Lisi et al. (2024)**: Up to **33 HU intra-individual variation** (34 HU inter-individual) between reconstruction kernels and iterative reconstruction levels. FAI values decrease with sharper kernels (Bv56: −106±2 HU vs smooth Qr36+QIR4: −87±9 HU). Increasing iterative reconstruction strength causes FAI to increase by up to 12 HU.
- The same patient reconstructed with different kernels can be classified as "inflamed" or "non-inflamed" depending on the reconstruction choice.

#### Scanner Platform
- **Tremamunno et al. (2025)**: Intra-individual FAI differences between PCD-CT and conventional EID CT confirmed measurements are NOT directly comparable.
- **Boussoussou et al. (2023)**: Average PCAT attenuation was **15 HU higher** on a different scanner (GE CardioGraphe vs Philips Brilliance) for the same patients.

#### Contrast Timing and Perfusion
- **Wu et al. (2025)**: ~**7 HU swing** in PCAT HU from contrast timing differences; ~**15% PCAT volume change**; **78% of radiomic features change >10%** between perfusion phases.

#### Body Habitus
- **Nie & Molloi (2025)**: 3.6% HU variance between small, medium, and large patient sizes for identical tissue
- **Boussoussou et al. (2023)**: BMI effect of **−0.4 HU per kg/m²**

#### Other Technical Confounders
- **Cardiac Phase**: Different cardiac phases (systole vs diastole) produce different apparent PCAT attenuation due to volumetric compression and motion.
- **Fat Threshold Selection**: Multiple studies use inconsistent fat threshold definitions (minimum: −200 to −149 HU; maximum: −45 to −30 HU), preventing direct cross-study comparison.
- **Partial Volume Effects**: Measurements within 0.75 mm of lumen are susceptible to partial volume effects from adjacent contrast-enhanced vessel.

### 4.2 The Partial Volume Problem

The argument that FAI primarily reflects partial volume effects rather than true biological changes challenges the entire biological rationale for FAI.

#### Hell 2016: Primary Partial Volume Argument
> Hell et al. *JCCT* 2016;10(1):52–60.

This study argued that variations in CT density of PCAT are primarily attributed to partial volume effects and image interpolation rather than tissue composition or metabolic activity:
- PCAT attenuation decreased with increasing distance from the vessel
- PCAT attenuation decreased from proximal to distal segments
- These patterns are consistent with partial volume contamination from the contrast-enhanced lumen rather than biological gradients

#### Distance-from-Lumen Gradient
- **Li et al. (2025)**: PCAT density is highest closest to lumen (within 0.5 mm) and decreases with distance. Measurements within 0.75 mm of lumen are particularly susceptible to partial volume effects.

#### Counterargument: Histological Validation
- **Antonopoulos et al. (2017)**: Histological validation directly measured tissue composition changes (lipid droplet size, transcription factors, cytokines) that matched CT-derived FAI — suggesting at least some of the signal is biological and not just partial volume artifact.

### 4.3 Negative and Contradictory Studies

Several studies have failed to replicate the FAI-plaque relationship, raising fundamental questions about FAI's reliability.

#### Boussoussou 2023: The Most Damaging Evidence
> Boussoussou et al. *JCCT* 2023. n=1,652 patients with zero calcium score (low-risk).

This is perhaps the most damaging study to the FAI thesis. Key findings:

| Analysis | NCP association with PCAT | p-value |
|---|---|---|
| **Univariable** | +2 HU with NCP presence | **<0.001** |
| **Multivariable** (corrected for imaging + patient factors) | **No association** | **0.93** |

After multivariable correction for patient and imaging characteristics, **NONE of the plaque markers remained associated with PCAT attenuation**.

Significant independent predictors of PCAT attenuation (not plaque, but technical/demographic):

| Factor | Effect on PCAT |
|---|---|
| Male sex | **+1 HU** |
| 120 kVp vs 100 kVp | **+8 HU** |
| Pixel spacing | **+32 HU per mm³** |
| Heart rate | **−0.2 HU per bpm** |
| BMI | **−0.4 HU per kg/m²** |
| Tube current | Significant |
| CNR / SNR | Significant |

**Implication**: The FAI-plaque relationship reported in many studies may be substantially or entirely confounded by imaging parameters and patient characteristics.

#### Pandey 2020: Counterintuitive Finding
> Pandey et al. *Br J Radiol* 2020;93:20200540.

Patients with obstructive CAD had even **lower** PCAT attenuation than those without — a counterintuitive finding suggesting the FAI signal is not as straightforward as "higher = more inflamed = more disease."

#### Other Negative Studies
- **Ma et al. (2021)**: Found no difference in PCAT attenuation between patients with and without CAD.
- **LoDoCo2 CT substudy**: Despite colchicine reducing MACE by 31%, PCAT attenuation did NOT change after 28 months (see §3.4 for full details).
- **Tan et al. (2025) meta-analysis**: Found inconsistent FAI methodologies across studies, high heterogeneity in measurements, and uncertain clinical predictive value for MACE in some subgroups.

### 4.4 The Standardisation Crisis

#### Tan JACC 2023 Review: "Not Ready for Prime Time"
> Tan et al. "Pericoronary Adipose Tissue as a Marker of Cardiovascular Risk." *JACC* 2023 (Review Topic of the Week).

This authoritative JACC review synthesises the state of the PCAT/FAI field and identifies critical unresolved issues:

**Key confounders highlighted** (with specific magnitudes):
- Tube voltage: **11 HU difference** between 70–120 kV
- Reconstruction algorithm: significant and variable effect
- Scanner type: not interchangeable
- Body habitus: BMI-dependent baseline shift
- Partial volume artifact: proximity to contrast-enhanced lumen
- Heart rate: affects motion and apparent attenuation

**Critical gaps identified**:
1. **No test-retest variability study**: Even within 24 hours with matched parameters, test-retest variability has not been formally quantified — a basic requirement for any clinical biomarker
2. **No head-to-head software comparison**: CaRi-Heart vs manual vs ShuKun vs other tools — never compared on the same dataset
3. **Optimal measurement site/extent undefined**: Proximal 40 mm (Oxford) vs lesion-specific (ShuKun) vs whole-vessel — no consensus
4. **RCA vs LAD vs LCX discordance**: Cannot reliably extrapolate from single artery to overall coronary inflammatory burden
5. **Discordance with high-risk plaque**: Figure 3 of the review shows 4 cases with different directional relationships between HRP and PCAT — they are NOT always concordant

**Verdict**: PCAT is a **promising but unvalidated** clinical biomarker that is **"not ready for prime time"** — needs standardisation, reproducibility validation, and prospective interventional trials before routine clinical deployment.

#### Chan & Antoniades 2025 Editorial Comment
> Chan & Antoniades 2025: "Pericoronary Adipose Tissue Imaging and the Need for Standardized Measurement of Coronary Inflammation" (**editorial comment**)

Acknowledged the standardisation problem and proposed the **"FAI Score"** — adjusting raw PCAT attenuation for technical factors (tube voltage, reconstruction), anatomical factors (vessel size, fat volume), and demographic factors (age, sex, BMI) to produce a standardised score. This is the Oxford group's own proposed solution, but it adds complexity and requires calibration datasets.

#### Němečková 2025 Review
> Němečková et al. 2025: "The Perivascular Fat Attenuation Index: Bridging Inflammation and Cardiovascular Disease Risk"

Comprehensive review highlighting the urgent need for standardisation across the field.

#### Current Solutions and Their Limitations

1. **Conversion factors (Etter et al.)**:
   - Partial solution — addresses kVp but not kernel or scanner effects
   - Requires phantom validation for each scanner model

2. **FAI Score (Antoniades)**:
   - Promising but requires large calibration datasets
   - Adds complexity to the measurement pipeline
   - Not yet independently validated across platforms

3. **Material decomposition (our approach)**:
   - Fundamentally protocol-independent — measures composition, not attenuation
   - Eliminates the need for calibration datasets
   - Directly addresses the root cause of the standardisation crisis

---

## Part V: Emerging Directions

### 5.1 PCAT Radiomics

#### Rationale: Beyond Mean HU

Mean HU (FAI) is a single summary statistic. The spatial distribution, texture, and heterogeneity of PCAT may contain additional prognostic information about the inflammatory microenvironment.

#### ShuKun 93-Feature Pipeline

> Huang et al. 2025, PMID 41163958 — Lesion-specific PCAT radiomics for MACE prediction

ShuKun's commercial pipeline extracts **93 radiomic features** per VOI:

| Feature Class | Count | Examples |
|---|---|---|
| First-order statistics | ~18 | Mean, median, energy, entropy, skewness, kurtosis, percentiles |
| GLCM (co-occurrence matrix) | ~24 | Contrast, correlation, entropy, homogeneity, cluster shade |
| GLSZM (size zone matrix) | ~16 | Small zone emphasis, large zone high grey level emphasis |
| GLRLM (run length matrix) | ~16 | Run length non-uniformity, long run emphasis |
| NGTDM (neighbourhood tone) | ~5 | Coarseness, complexity, busyness |
| GLDM (dependence matrix) | ~14 | Dependence variance, dependence entropy |

ML pipeline: Pearson correlation filtering → Lasso (L1) → XGBoost with 10-fold CV → MACE prediction.

#### Key Radiomics Studies

- **Shang et al. (2025, *Cardiovasc Diabetol*, n=777, multicentre)**: PCAT radiomics improved MACE prediction beyond traditional risk scores. Combined model C-index: 0.873 (training), 0.824 (validation). Significant reclassification improvement (NRI: 0.256–0.480).
- **Hou/Liu et al. (2024, *Insights Imaging*, n=180)**: PCAT radiomic signature predicted rapid plaque progression. First-order statistics and higher-order texture features were most predictive.
- **Huang et al. (2025, *Front Cardiovasc Med*)**: Compared lesion-specific vs proximal PCAT radiomics models for MACE — lesion-specific approach showed superior performance.
- **PCAT radiomics for INOCA in NAFLD (2025, *BMC Cardiovasc Disord*)**: Radiomics model (AUC 0.734) outperformed simple PCAT attenuation (AUC 0.674) for diagnosing ischaemia with non-obstructive coronary arteries in NAFLD patients.

#### Critical Limitation: Radiomic Feature Instability

> Wu et al. (2025, Case Western): **78% of radiomic features change >10%** between different contrast perfusion phases.

This means radiomic models trained on one acquisition timing may not generalise to different timing protocols — the same protocol-dependence problem as FAI, but amplified across 93 features.

### 5.2 Photon-Counting CT (PCD-CT)

#### What PCD-CT Offers

PCD-CT (Siemens NAEOTOM Alpha, GE Revolution CT) provides simultaneous multi-energy data:
- VMI at any keV (40–190 keV)
- Material decomposition maps (water, iodine, lipid, calcium)
- Effective atomic number (Z-eff) maps
- Ultra-high resolution mode (0.2 mm pixels)

#### Key PCD-CT Studies

- **Mergen et al. (2021, *AJR*)**: First systematic assessment of EAT/FAI on PCD-CT. Phantom and in vivo (n=30). VMI at 55–80 keV compared to reference 120 kVp EID scan. Fat attenuation varies significantly with VMI energy level — 70 keV VMI approximates 120 kVp but is not identical.
- **Tremamunno et al. (2025, *Acad Radiol*)**: Intra-individual FAI differences between PCD-CT and conventional EID CT confirmed measurements are NOT directly comparable. However, iterative reconstruction minimises most differences, enabling inter-scanner comparability.
- **Kravchenko et al. (2025, *Int J Cardiol*)**: Extended the Tremamunno FAI work to full radiomic analysis — compared PCAT radiomic feature stability between PCD-CT and EID-CT within the same patients.
- **Kahmann et al. (2024, *Front Cardiovasc Med*)**: PCAT texture analysis and CAD characterization on PCD-CT. Explored radiomic features of LAD and RCA PCAT on the NAEOTOM Alpha platform.
- **Engel et al. (2026, *J Clin Med*)**: First study applying the −70.1 HU FAI threshold on PCD-CT. FAI ≥ −70.1 HU identified more lipid-rich, non-calcified plaques (vulnerable morphology).
- **Gao et al. (2025, *Eur J Radiol*)**: PCD-CT UHR mode significantly reduced stent blooming artifacts. Stent-specific FAI was lower in PCD-CT vs simulated conventional CT.

#### VMI Considerations

VMI at 70 keV closely matches conventional 120 kVp CT in noise characteristics, but fat HU values shift approximately **+5 to +15 HU** due to energy-dependent attenuation differences. The −70.1 HU threshold has not been validated on VMI data.

#### New Threshold Development

- **2025 (*Front Cardiovasc Med*)**: Development of NEW threshold for pericoronary fat attenuation based on **40 keV VMI** from dual-energy spectral CT — recognizing the old threshold doesn't transfer.

### 5.3 Material Decomposition — Our Approach

#### Core Argument

All of the confounders described in Part IV affect **HU values** (the physical measurement underlying FAI) but do NOT affect the **actual tissue composition** (water, lipid, protein, iodine content). Material decomposition — decomposing each voxel into its constituent materials — is inherently protocol-independent because it measures composition, not attenuation.

#### Key Results from Nie & Molloi 2025

> Nie S, Molloi S. "Quantification of Water and Lipid Composition of Perivascular Adipose Tissue Using Coronary CT Angiography: A Simulation Study." *Int J Cardiovasc Imaging* 2025;41:1091–1101.

- Water fraction RMSE: 0.01–0.64% (sufficient to detect the ~5% clinical threshold)
- HU variance across 80–135 kV: **21.9%** (protocol-dependent)
- HU variance across patient sizes: **3.6%** (protocol-dependent)
- **Material decomposition (water fraction) was protocol-independent**: same composition yielded same water fraction regardless of kV or patient size

#### Current XCAT Study

The current study extends this work by using anatomically realistic XCAT phantoms, simulating pericoronary adipose inflammation as increased water content, and decomposing into 4 materials (water, lipid, collagen, iodine). The key demonstration: FAI (HU) differs across protocols for the same tissue, but material decomposition gives consistent composition regardless of protocol.

#### Why This Matters for the Field

For multi-site trials, longitudinal monitoring, and cross-scanner comparisons, a protocol-independent biomarker is essential. The field is increasingly recognising this need (§4.4), and material decomposition provides a fundamentally different approach than the calibration/correction strategies (conversion factors, FAI Score) currently proposed.

---

## Part VI: Field Landscape

### 6.1 Research Groups

| Group | Location | Key Investigators | Primary Contributions | Key Papers |
|---|---|---|---|---|
| **Oxford/Antoniades** | UK | Charalambos Antoniades, Evangelos Oikonomou, Kenneth Chan | Defined FAI, CRISP-CT and ORFAN trials, CaRi-Heart commercialisation | Antonopoulos *Sci Transl Med* 2017; Oikonomou *Lancet* 2018; Oikonomou *Nat CV Res* 2023; Chan *Lancet* 2024 |
| **Erlangen/Achenbach** | Germany | Stephan Achenbach, Mohamed Marwan | CRISP-CT derivation cohort, early PCD-CT FAI studies | Oikonomou *Lancet* 2018; Engel *J Clin Med* 2026 |
| **Cedars-Sinai/Monash** | USA/Australia | Damini Dey, Andrew Lin, Daniel Berman, Stephen Nicholls, Dennis Wong | PCAT radiomics, ML models, statin effects, CT-FFR integration | Multiple 2019–2025 on PCAT radiomics |
| **Zurich/Alkadhi** | Switzerland | Hatem Alkadhi, Katharina Eberhard, André Mergen | PCD-CT FAI evaluation, kernel/reconstruction effects | Eberhard 2022–2025; Mergen 2022–2025; Lisi *Eur Radiol* 2024 |
| **Groningen/Vliegenthart** | Netherlands | Rozemarijn Vliegenthart, Riemer Ma | PCAT reference values per vessel per kV, low-kV effects | Ma et al. 2020 |
| **Case Western/Rajagopalan** | USA | Sanjay Rajagopalan, Chris Wu, David Wilson | Quantified PCAT confounds — contrast timing, radiomic instability | Wu et al. 2025 |
| **Tohoku/Shimokawa** | Japan | Hiroaki Shimokawa, Kensuke Ohyama | PVAT inflammation in vasospastic angina using 18F-FDG PET | Ohyama et al. 2016–2017 |
| **Korean Groups** | South Korea | Multiple (SNU, Asan, Samsung, Yonsei) | ICONIC study, lesion-specific PCAT, CCTA-OCT, ethnic validation | Multiple Korean collaborations |
| **Chinese Groups** | China | Multiple (Fudan, Beijing Anzhen, West China, Guangdong) | Large multicentre cohorts, PCAT radiomics, special populations | Shang *Cardiovasc Diabetol* 2025; Lu *BMC Nephrol* 2025 |
| **Japanese Groups** | Japan | Multiple (Kobe, Okayama, Kawasaki) | East Asian validation, periprocedural injury, T2DM cohorts | Multiple Japanese collaborations |
| **European Groups** | Europe | Multiple (Amsterdam, Mannheim, Leiden, Milano) | CaRi-Heart deployment, PCD-CT radiomics, innovative imaging | Multiple European collaborations |
| **MolloiLab/UCI** | USA | Sabee Molloi, Shu Nie | Material decomposition for coronary plaque (DECT), water-lipid-protein PVAT decomposition | Ding et al. 2021; Nie & Molloi *Int J Cardiovasc Imaging* 2025 |

### 6.2 Commercial Landscape

| Company | FAI Analysis | Plaque Analysis | AI Risk Score | FDA Cleared | Reimbursement |
|---|---|---|---|---|---|
| **Caristo (CaRi-Heart)** | ✅ Core product | ✅ CaRi-Plaque | ✅ CaRi-Heart score | ✅ (2025) | ✅ Medicare 2026 |
| **Cleerly** | ❌ | ✅ Core product | ✅ | ✅ | ✅ Category I CPT |
| **HeartFlow** | ❌ | ✅ | ✅ (with FFR) | ✅ | ✅ Category I CPT |
| **ShuKun** | ✅ Radiomics | ✅ | Partial | Regional (China) | Regional |

#### Caristo Diagnostics (Oxford Spinoff)
- **Product**: CaRi-Heart (FAI-based risk) + CaRi-Plaque™ (AI plaque analysis)
- **FDA 510(k) clearance**: CaRi-Plaque™ (March 2025)
- **CPT codes**: AMA assigned Category III codes 0992T and 0993T (2025)
- **Medicare reimbursement**: Finalized across hospital and office settings starting 2026
- **Clinical deployment**: First U.S. hospital (NCH) implementing CaRi-Heart
- **NHS study (2024)**: Demonstrated CaRi-Heart could reduce cardiac deaths by 12% in the UK NHS

#### Cleerly
- **Focus**: AI-based comprehensive coronary plaque analysis (not specifically FAI-focused)
- **Funding**: $106M round (2024)
- **CPT**: Category I code for AI-QCT advanced plaque analyses
- **Coverage**: Aetna, UnitedHealthcare, Cigna, Humana — 86+ million lives
- **Distinction**: Emphasises plaque characterisation (stenosis, composition, remodelling) over pericoronary inflammation

#### HeartFlow
- **Product**: HeartFlow Plaque Analysis, FDA 510(k) cleared (2025)
- **CPT**: New Category I code for AI-enabled plaque analysis
- **Funding**: $890.5M total
- **Focus**: CT-FFR + plaque analysis, not specifically PCAT/FAI

#### ShuKun Technology (China)
- **Product**: Peri-coronary Adipose Tissue Analysis Tool + CoronaryDoc®-FFR
- **Funding**: $296.4M total
- **Focus**: PCAT radiomics (93 features), lesion-specific analysis
- **Market**: Strong presence in Asian markets, used in multiple Chinese multicentre studies

### 6.3 Research Trajectory Timeline

| Year | Milestone | Group | Key Paper |
|---|---|---|---|
| 1999 | Atherosclerosis defined as inflammatory disease | Ross | Ross R, *NEJM* 1999 |
| 2005 | PVAT macrophage infiltration characterised | Henrichot et al. | *ATVB* 2005 |
| 2007–2015 | Vasa vasorum role in atherosclerosis established | Ritman & Lerman (Mayo) | Multiple reviews |
| 2016 | PCAT density: partial volume effects argument | Hell, Achenbach (Erlangen) | Hell et al., *JCCT* 2016 |
| 2016–2017 | Vasospastic angina linked to PVAT inflammation (PET) | Shimokawa (Tohoku) | Ohyama et al. |
| 2017 | CANTOS trial: IL-1β causal role in MACE | Ridker (Brigham) | *NEJM* 2017 |
| 2017 | FAI concept introduced with histological validation (n=453) | Antoniades (Oxford) | Antonopoulos et al., *Sci Transl Med* |
| 2018 | **CRISP-CT**: FAI validated as prognostic biomarker (HR 9.04) | Oxford + Erlangen | Oikonomou et al., *Lancet* |
| 2019 | COLCOT: colchicine −23% MACE post-MI | Tardif (Montreal) | *NEJM* 2019 |
| 2020 | LoDoCo2: colchicine −31% MACE in chronic CAD | Nidorf | *NEJM* 2020 |
| 2020 | PCAT reference values established per vessel, per kV (n=493) | Groningen | Ma et al. 2020 |
| 2021 | Material decomposition for coronary plaque (DECT) | Molloi (UCI) | Ding et al. 2021 |
| 2022 | Meta-analysis: FAI in unstable vs stable plaques (n=7,797) | Sagris et al. | Sagris et al. 2022 |
| 2022 | Phantom study: PCATMA affected by kVp and reconstruction | Etter et al. | Etter et al. 2022 |
| 2022–2024 | PCD-CT FAI studies begin | Zurich, Mannheim | Multiple |
| 2023 | ORFAN: CaRi-Heart AI-FAI outperforms conventional risk (n=3,324) | Oxford | *Nat CV Res* 2023 |
| 2023 | Tan JACC review: PCAT "not ready for prime time" | Baker Heart/Cedars-Sinai | Tan et al., *JACC* 2023 |
| 2023 | Boussoussou: No PCAT-CAD correlation after adjustments | Semmelweis/Cedars-Sinai | Boussoussou et al. 2023 |
| 2024 | ORFAN extended: n=40,091, 7.7 yr follow-up | Oxford/multicentre | Chan et al., *Lancet* 2024 |
| 2024 | Lisi: kernel effects up to 33 HU variation | Zurich | *Eur Radiol* 2024 |
| 2024 | PCD-CT vs EID: FAI not comparable | Tremamunno, Schoepf (MUSC) | *Acad Radiol* 2025;32(3) |
| 2025 | Water-lipid-protein for PVAT (simulation) | Nie S, Molloi (UCI) | *Int J Cardiovasc Imaging* 2025 |
| 2025 | Wu: perfusion confounds (7 HU swing, 78% radiomic instability) | Case Western | Wu et al. 2025 |
| 2025 | Caristo: FDA clearance CaRi-Plaque, CPT codes, Medicare | Oxford/Caristo | — |
| 2025 | LoDoCo2 CT substudy: colchicine does NOT change FAI | Fiolet et al. | *Heart* 2025 |
| 2025 | PCAT radiomics multicentre (n=777) | Shang et al. (China) | *Cardiovasc Diabetol* 2025 |
| 2025 | FAI Score standardization proposed (editorial comment) | Chan & Antoniades | Oxford |
| 2026 | **Current study**: XCAT + material decomposition for PCAT | Nie S, Molloi (UCI) | *In preparation* |

---

## Part VII: Methodology Patterns

### 7.1 Study Design Patterns

| Study Type | Proportion | Examples |
|---|---|---|
| **Retrospective cohort** | ~60% | CRISP-CT, most single-centre PCAT studies |
| **Prospective cohort** | ~15% | ORFAN, some NAEOTOM Alpha studies |
| **Meta-analysis / systematic review** | ~10% | Sagris et al. 2022, Tan et al. 2025 |
| **Phantom / simulation** | ~10% | Etter et al. 2022, Nie & Molloi 2025 |
| **Case-control** | ~5% | ACS vs. stable angina comparisons |

### 7.2 PCAT Measurement Protocol

Nearly all clinical PCAT studies follow the **Oxford/CRISP-CT protocol**:
- Fat HU window: −190 to −30 HU
- VOI: outer vessel wall + 1× mean vessel diameter, proximal 40 mm (LAD/LCX) or 10–50 mm (RCA)
- Primary metric: Mean HU of fat-range voxels = FAI
- Threshold: −70.1 HU (high-risk)

**Variations**:
- **Lesion-specific PCAT** (ShuKun): VOI around individual plaques, not fixed proximal segments
- **Volumetric PCAT**: Total fat-range voxel volume in cm³ (complementary to FAI)
- **Radiomic PCAT**: 93-feature extraction per VOI (GLCM, GLRLM, GLSZM, NGTDM, GLDM)
- **PCATMA**: No fat threshold — measures all tissue attenuation in the pericoronary VOI

### 7.3 Patient Selection

#### Common Inclusion Criteria

| Criterion | Typical Requirement | Rationale |
|---|---|---|
| Indication | Clinically indicated CCTA for suspected/known CAD | Ensure clinical relevance |
| Image quality | Adequate for coronary assessment (motion score ≤ 2) | FAI requires clear vessel-fat boundary |
| Contrast enhancement | Adequate opacification (aortic root >250 HU) | Ensure proper contrast timing |
| ECG gating | Successful gating with evaluable phase | Motion-free reconstruction |
| Age | Typically >18 years, often >40 years | CAD prevalence |

#### Common Exclusion Criteria

| Criterion | Rationale |
|---|---|
| Prior CABG or coronary stenting in target vessel | Metal artifact contaminates VOI |
| Severe coronary calcification (Agatston >1000) | Blooming artifact affects adjacent fat HU |
| Anomalous coronary anatomy | VOI construction assumes normal anatomy |
| Severe motion artifact | Unreliable fat-vessel boundary |
| BMI extremes (>40 or <18) | Body habitus affects image quality and HU calibration |
| Active systemic infection or autoimmune disease | Confounds inflammatory signal |
| Recent cardiac surgery (<3 months) | Post-surgical inflammation confounds |

### 7.4 Patient Data in Key Studies

| Study | n | Selection | Follow-up | Primary Endpoint |
|---|---|---|---|---|
| Antonopoulos 2017 | 453 | Cardiac surgery with biopsies | Cross-sectional | Histological validation |
| CRISP-CT 2018 | 1,872 | Clinically indicated CCTA | 5 years | Cardiac death (HR 9.04) |
| ORFAN 2023 | 3,324 | Prospective CCTA cohort | Ongoing | MACE |
| ORFAN extended 2024 | 40,091 | Multicentre CCTA | 7.7 years | MACE + cardiac mortality |
| Sagris meta 2022 | 7,797 | 20 studies pooled | Variable | Unstable vs stable plaque |
| Ma et al. 2020 | 493 | Consecutive CCTA, no known CAD | Cross-sectional | Reference values |
| Shang et al. 2025 | 777 | 3 centres, ACS patients | 5.45 years | MACE (radiomics) |
| Lu et al. 2025 | 444 | CAD + CKD | 4.57 years | MACE + CV mortality |
| Wu et al. 2025 | 135 | CT perfusion patients | Cross-sectional | Perfusion timing confounds |
| Lisi et al. 2024 | 100 | Reconstruction comparison | Cross-sectional | Kernel effect on FAI |
| Coerkamp 2025 | 50 | High-risk patients | Cross-sectional | Risk reclassification |
| Fiolet 2025 | 151 | LoDoCo2 substudy | 28 months | Colchicine effect on FAI |
| Boussoussou 2023 | 1,652 | Zero calcium score | Cross-sectional | PCAT confounders |

---

## References

1. Ross R. "Atherosclerosis — an inflammatory disease." *NEJM* 1999;340:115–126. [DOI](https://doi.org/10.1056/NEJM199901143400207)
2. Libby P et al. "The changing landscape of atherosclerosis." *Circulation* 2021. [DOI](https://doi.org/10.1161/CIRCULATIONAHA.121.054137)
3. Ridker PM et al. "Antiinflammatory therapy with canakinumab for atherosclerotic disease (CANTOS)." *NEJM* 2017;377:1119–1131. [DOI](https://doi.org/10.1056/NEJMoa1707914)
4. Tardif JC et al. "Efficacy and safety of low-dose colchicine after myocardial infarction (COLCOT)." *NEJM* 2019;381:2497–2505. [DOI](https://doi.org/10.1056/NEJMoa1912388)
5. Nidorf SM et al. "Colchicine in patients with chronic coronary disease (LoDoCo2)." *NEJM* 2020;383:1838–1847. [DOI](https://doi.org/10.1056/NEJMoa2021372)
6. Antonopoulos AS et al. "Detecting human coronary inflammation by imaging perivascular fat." *Sci Transl Med* 2017;9:eaal2658. [DOI](https://doi.org/10.1126/scitranslmed.aal2658)
7. Oikonomou EK et al. "Non-invasive detection of coronary inflammation using CT and prediction of residual cardiovascular risk (CRISP-CT)." *Lancet* 2018;392:929–939. [DOI](https://doi.org/10.1016/S0140-6736(18)31114-0)
8. Oikonomou EK et al. "A novel machine learning-derived radiotranscriptomic map of perivascular biology (ORFAN)." *Nat Cardiovasc Res* 2023. [DOI](https://doi.org/10.1038/s44161-023-00246-8)
9. Chan K, Wahome SKW, Antoniades C et al. "Inflammatory risk and cardiovascular events in patients without obstructive coronary artery disease (ORFAN extended)." *Lancet* 2024. [DOI](https://doi.org/10.1016/S0140-6736(24)01811-9)
10. Fiolet ATL et al. "Effect of low-dose colchicine on pericoronary inflammation and coronary plaque composition (LoDoCo2 CT substudy)." *Heart* 2025. [DOI](https://doi.org/10.1136/heartjnl-2024-325527)
11. Ma R et al. "Towards reference values of pericoronary adipose tissue attenuation." *Eur Radiol* 2020. [DOI](https://doi.org/10.1007/s00330-020-07069-0)
12. Sagris M et al. "Pericoronary fat attenuation index — meta-analysis." *Eur Heart J Cardiovasc Imaging* 2022. [DOI](https://doi.org/10.1093/ehjci/jeac174)
13. Wu C et al. "Perfusion confounds on pericoronary adipose tissue." *J Clin Med* 2025. [DOI](https://doi.org/10.3390/jcm14030769)
14. Lisi C et al. "Kernel and reconstruction effects on FAI." *Eur Radiol* 2024. [DOI](https://doi.org/10.1007/s00330-024-11132-5)
15. Etter M et al. "Phantom kVp study — PCATMA conversion factors." *Eur Radiol* 2022. [DOI](https://doi.org/10.1007/s00330-022-09274-5)
16. Nie S, Molloi S. "Quantification of water and lipid composition of perivascular adipose tissue using coronary CT angiography: a simulation study." *Int J Cardiovasc Imaging* 2025;41:1091–1101. [DOI](https://doi.org/10.1007/s10554-025-03358-5)
17. Engel et al. "FAI on photon-counting CT." *J Clin Med* 2026. [DOI](https://doi.org/10.3390/jcm15010140)
18. Shang J et al. "PCAT radiomics multicentre study." *Cardiovasc Diabetol* 2025. [DOI](https://doi.org/10.1186/s12933-025-02913-3)
19. Lu et al. "FAI in CAD with CKD." *BMC Nephrol* 2025. [DOI](https://doi.org/10.1186/s12882-025-04549-7)
20. Coerkamp et al. "FAI reclassifies 62% of patients." *Int J Cardiol Cardiovasc Risk Prev* 2025. [DOI](https://doi.org/10.1016/j.ijcrp.2024.200360)
21. Li et al. "FAI for culprit lesion identification in ACS." *J Comput Assist Tomogr* 2025. [DOI](https://doi.org/10.1097/RCT.0000000000001704)
22. Li et al. "PCATMA vs FAI diagnostic comparison." *Quant Imaging Med Surg* 2025. [DOI](https://doi.org/10.21037/qims-24-828)
23. Zhang et al. "FAI and diabetes duration." *Front Endocrinol* 2025. [DOI](https://doi.org/10.3389/fendo.2025.1671949)
24. Yuasa et al. "FAI predicts HFpEF hospitalization." *JACC Advances* 2025. [DOI](https://doi.org/10.1016/j.jacadv.2025.101685)
25. Diau & Lange. "Inflammation in non-obstructive CAD and MINOCA." *Curr Cardiol Rep* 2025. [DOI](https://doi.org/10.1007/s11886-025-02221-y)
26. Němečková et al. "The perivascular fat attenuation index: bridging inflammation and cardiovascular disease risk." *J Clin Med* 2025. [DOI](https://doi.org/10.3390/jcm14134753)
27. Tremamunno G et al. "Intra-individual differences in pericoronary FAI between PCD and EID CT." *Acad Radiol* 2025;32(3). [DOI](https://doi.org/10.1016/j.acra.2024.11.055)
28. Chan & Antoniades. "Pericoronary adipose tissue imaging and the need for standardized measurement of coronary inflammation." 2025. (**editorial comment**) [DOI](https://doi.org/10.1093/eurheartj/ehaf012)
29. Ohyama K et al. "PVAT inflammation in vasospastic angina." *Circ J* 2016. [DOI](https://doi.org/10.1253/circj.CJ-16-0213)
30. Moser PT et al. "FAI in cardiac transplant." *Eur Radiol* 2023. [DOI](https://doi.org/10.1007/s00330-023-09614-z)
31. Baritussio A et al. "FAI in myocarditis." *J Clin Med* 2021. [DOI](https://doi.org/10.3390/jcm10184200)
32. Boussoussou et al. "The effect of patient and imaging characteristics on CCTA-assessed pericoronary adipose tissue." *JCCT* 2023. [DOI](https://doi.org/10.1016/j.jcct.2022.09.006)
33. Huang et al. "ShuKun PCAT radiomics for MACE." *Front Cardiovasc Med* 2025. PMID: 41163958. [DOI](https://doi.org/10.3389/fcvm.2025.1600942)
34. Hou/Liu et al. "PCAT radiomic signature predicts plaque progression." *Insights Imaging* 2024. [DOI](https://doi.org/10.1186/s13244-024-01731-7)
35. Iacobellis G. "Local and systemic effects of the multifaceted epicardial adipose tissue depot." *Nat Rev Endocrinol* 2015. [DOI](https://doi.org/10.1038/nrendo.2015.58)
36. Henrichot E et al. "Production of chemokines by perivascular adipose tissue." *ATVB* 2005. [DOI](https://doi.org/10.1161/01.ATV.0000188508.40052.35)
37. Ding Y, Molloi S. "Material decomposition for coronary plaque using DECT." *Int J Cardiovasc Imaging* 2021. [DOI](https://doi.org/10.1007/s10554-024-03124-9)
38. Hell MM, Achenbach S et al. "CT-based analysis of pericoronary adipose tissue density." *JCCT* 2016;10(1):52–60. [DOI](https://doi.org/10.1016/j.jcct.2015.07.011)
39. Pandey NN et al. "Epicardial fat attenuation, not volume, predicts obstructive CAD." *Br J Radiol* 2020;93:20200540. [DOI](https://doi.org/10.1259/bjr.20200540)
40. Tan N et al. "Pericoronary adipose tissue as a marker of cardiovascular risk." *JACC* 2023 (Review Topic of the Week). [DOI](https://doi.org/10.1016/j.jacc.2022.12.021)
41. Kravchenko D, Tremamunno G et al. "Intra-individual radiomic analysis of pericoronary adipose tissue: PCD vs EID CT." *Int J Cardiol* 2025;420:132749. [DOI](https://doi.org/10.1016/j.ijcard.2025.132749)
42. Kahmann J, Ayx I et al. "Interrelation of pericoronary adipose tissue texture and CAD on photon-counting CT." *Front Cardiovasc Med* 2024;11:1499219. [DOI](https://doi.org/10.3389/fcvm.2024.1499219)
43. Mergen V, Eberhard M, Alkadhi H et al. "Epicardial adipose tissue attenuation and FAI: phantom study and in vivo measurements with photon-counting detector CT." *AJR* 2021. [DOI](https://doi.org/10.2214/AJR.21.26930)