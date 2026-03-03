# Coronary Artery Inflammation and Pericoronary Adipose Tissue: Comprehensive Field Review

**Project**: PCAT Segmentation Pipeline — MolloiLab  
**Author**: Shu Nie  
**Date**: March 2026  
**Scope**: In-depth review of the PCAT/FAI field — clinical motivations, biological basis, landmark trials, prognostic evidence, research groups, study design patterns, patient selection, technical limitations, emerging applications, PCAT radiomics, photon-counting CT, commercial landscape, and open questions. Synthesised from ~85 papers in the coronary inflammation collection, online literature search (2023–2026), and cross-referencing of major reviews.

---

## Table of Contents

1. [Clinical Motivations](#1-clinical-motivations-for-investigating-pcat)
2. [Atherosclerosis as Inflammatory Disease](#2-paradigm-shift-atherosclerosis-as-an-inflammatory-disease)
3. [PCAT as Paracrine Signalling Hub](#3-perivascular-adipose-tissue-as-a-paracrine-signalling-hub)
4. [PCAT vs. EAT](#4-pcat-vs-epicardial-adipose-tissue-eat)
5. [Spatial Heterogeneity](#5-spatial-heterogeneity-of-coronary-inflammation)
6. [Landmark Clinical Trials](#6-landmark-clinical-trials-and-evidence-base)
7. [FAI Prognostic Evidence (2023–2026)](#7-recent-fai-prognostic-evidence-20232026)
8. [FAI Beyond Atherosclerosis](#8-fai-beyond-atherosclerosis-emerging-applications)
9. [PCAT Radiomics](#9-pcat-radiomics-beyond-mean-hu)
10. [Technical Confounders and Limitations](#10-technical-confounders-and-limitations-of-fai)
11. [Negative Studies and Criticisms](#11-negative-studies-and-criticisms)
12. [Photon-Counting CT and Spectral Imaging](#12-photon-counting-ct-and-spectral-imaging)
13. [Commercial Landscape](#13-commercial-landscape)
14. [Research Groups](#14-major-research-groups)
15. [Research Trajectory Timeline](#15-research-trajectory-timeline)
16. [Study Design Patterns](#16-study-design-patterns-in-pcat-research)
17. [Patient Selection Methods](#17-patient-selection-methods)
18. [Material Decomposition — Our Approach](#18-material-decomposition--our-approach)
19. [Key References](#19-key-references)

---

## 1. Clinical Motivations for Investigating PCAT

### 1.1 The Residual Cardiovascular Risk Problem

Despite optimal lipid-lowering therapy with statins, a substantial **residual cardiovascular risk** persists. The CANTOS trial (Ridker et al., *NEJM* 2017, n=10,061) proved that this residual risk is driven by **inflammation independent of LDL cholesterol**: canakinumab (IL-1β antibody) reduced MACE by 15% without affecting LDL levels. This established inflammation as a direct, causal therapeutic target.

The colchicine trials subsequently confirmed a practical, affordable anti-inflammatory pathway:
- **COLCOT** (Tardif et al., *NEJM* 2019): −23% MACE post-MI
- **LoDoCo2** (Nidorf et al., *NEJM* 2020): −31% MACE in chronic CAD

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

## 2. Paradigm Shift: Atherosclerosis as an Inflammatory Disease

### 2.1 The Historical View (Lipid-Centric)

Until the 1990s, atherosclerosis was conceptualised primarily as a **lipid storage disorder**: LDL cholesterol accumulates in the arterial intima, forms fatty streaks, and progressively obstructs the lumen. While lipid lowering reduces MACE by ~35%, a large residual cardiovascular risk remains even after optimal statin therapy.

### 2.2 The Inflammatory Hypothesis

> Ross R. "Atherosclerosis — an inflammatory disease." *NEJM*. 1999;340:115–126.

Ross established the **"response-to-injury" hypothesis**: the primary trigger is endothelial injury (from oxidised LDL, hemodynamic shear stress, hypertension, smoking), which activates an inflammatory cascade:

1. Endothelial activation → upregulation of adhesion molecules (VCAM-1, ICAM-1, E-selectin)
2. Monocyte recruitment → differentiation into macrophages
3. Macrophages engulf oxidised LDL → foam cells
4. Foam cells secrete pro-inflammatory cytokines (IL-1β, IL-6, TNF-α)
5. Smooth muscle cell migration and proliferation → fibrous cap formation
6. Plaque vulnerability determined by cap thickness vs. inflammatory burden

### 2.3 Causal Proof: The CANTOS Trial

> Ridker PM et al. *NEJM* 2017;377:1119–1131. n=10,061.

CANTOS enrolled patients with prior MI and elevated hsCRP (≥2 mg/L). Canakinumab (IL-1β antibody) or placebo was given on top of optimal statin therapy:

- **Primary result**: MACE reduced by **15%** at 150 mg dose (p=0.031)
- Effect was **independent of LDL cholesterol** (LDL did not change)
- Dose-dependent reduction in hsCRP, IL-6
- First proof that targeting inflammation — not lipids — reduces cardiovascular events

### 2.4 The Inflammasome Pathway

> Libby P et al. *Circulation* 2021; *Nature* 2021.

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

## 3. Perivascular Adipose Tissue as a Paracrine Signalling Hub

### 3.1 Anatomy

**Pericoronary adipose tissue (PCAT)** is the adipose tissue immediately surrounding the coronary arteries, between the coronary vessels and the epicardial surface:
- Distinct from **epicardial adipose tissue (EAT)**: the full fat depot within the pericardial sac
- Distinct from **paracardial fat**: fat outside the pericardium
- Lacks a fascial barrier between the fat and the adventitia → direct exchange pathway
- Vascularised by the vasa vasorum, sharing the same microvascular supply as the adventitia

### 3.2 Vasocrine Signalling: Vessel Wall → Fat

When the coronary artery is inflamed:

1. Adventitial macrophages and smooth muscle cells secrete **IL-6, TNF-α, CXCL10, VEGF**
2. These mediators diffuse outward into adjacent PCAT
3. IL-6 and TNF-α suppress adipocyte differentiation:
   - Inhibit **PPARγ** (master transcription factor for fat cell maturation)
   - Inhibit **C/EBPα** (co-factor for adipogenesis)
   - Inhibit **FABP4** (fatty acid binding protein, marker of mature adipocyte)
4. Adipocytes remain immature, smaller, with less stored lipid
5. **Result on CT**: fat voxels shift from lipid-dominant (HU ≈ −90 to −70) toward more aqueous composition (HU ≈ −60 to −40) — this is the FAI signal

The FAI increase reflects a **genuine molecular phenotypic shift** in the fat cells, validated histologically by Antonopoulos et al. (2017, *Science Translational Medicine*, n=453 cardiac surgery patients): perivascular fat sampled adjacent to inflamed coronary segments showed reduced lipid droplet size (p<0.001), reduced PPARγ (−2.3-fold), reduced FABP4 (−1.8-fold), and increased IL-6 (+3.1-fold) and TNF-α (+2.7-fold).

### 3.3 Paracrine Signalling: Fat → Vessel Wall

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

### 3.4 Vasa Vasorum and Microvascular Inflammation

The **vasa vasorum** — the microvascular network supplying the coronary arterial wall — plays a key role in PCAT inflammation. Studies using micro-CT and histology (Kwon et al. 2015; Ritman & Lerman 2007) have shown:

- Neovascularisation of vasa vasorum (adventitial angiogenesis) precedes and promotes atherosclerotic plaque development
- Inflamed vasa vasorum provide the route for inflammatory cell infiltration into the vessel wall and adjacent PCAT
- PCAT surrounding vessels with dense vasa vasorum networks shows higher HU (more inflammation)

---

## 4. PCAT vs. Epicardial Adipose Tissue (EAT)

| Feature | PCAT | EAT |
|---|---|---|
| **Spatial scope** | ~3–5 mm shell around specific vessel segment | Entire pericardial sac |
| **Measurement unit** | Mean HU (attenuation = FAI) | Total volume (cm³) |
| **Clinical signal** | Per-vessel acute inflammatory state | Whole-heart chronic metabolic risk |
| **Segmentation** | Vessel-specific VOI (centerline-based) | Pericardial sac segmentation |
| **Key paper** | Oikonomou 2018, *Lancet* | Iacobellis, multiple reviews |
| **Commercial tool** | CaRi-Heart (Caristo), ShuKun PCAT | Multiple EAT volume tools |

Correlation between EAT volume and RCA-FAI is moderate at best (r ≈ 0.3–0.4). A patient with high EAT volume but low FAI is metabolically obese but coronary arteries not actively inflamed; low EAT volume but high FAI indicates lean but with focal plaque-driven coronary inflammation. Both are independently associated with MACE.

---

## 5. Spatial Heterogeneity of Coronary Inflammation

### 5.1 Why Proximal Segments Matter

The proximal coronary segments (LAD 0–40 mm, LCX 0–40 mm, RCA 10–50 mm) are the clinical focus because:
1. **Most plaques form here**: hemodynamic shear stress at bifurcations and proximal curves
2. **Clinical consequence**: proximal stenoses cause more downstream ischemia
3. **CT resolution**: proximal vessels are larger (3–5 mm diameter) → better SNR for FAI

### 5.2 Lesion-Specific PCAT

Rather than fixed proximal segments, measuring PCAT adjacent to each individual plaque provides:
- **Higher specificity**: PCAT around a stable fibrous plaque may be low even in the same vessel as an unstable plaque with high PCAT
- **Better MACE prediction**: the PCAT signal is strongest immediately adjacent to the vulnerable plaque (Huang et al. 2025)

Li et al. (2025, *QIMS*) compared PCATMA (Pericoronary Adipose Tissue Mean Attenuation — not constrained by fat threshold) with FAI and found PCATMA showed significant differences for non-calcified plaque (P<0.001) and mixed plaque (P=0.047), while FAI did not — suggesting threshold-free measurement may be more sensitive.

### 5.3 RCA vs. LAD vs. LCX: Different Biology

The clinical literature focuses primarily on RCA-FAI because:
- RCA has the most pericoronary fat (largest fat depot, cleaner VOI)
- LAD runs in the anterior interventricular groove — also well-studied
- LCX runs in the atrioventricular groove adjacent to the left atrial wall → VOI contamination more common

Reference values (Ma et al. 2020, Groningen, n=493): LAD −92.4 HU, LCX −88.4 HU, RCA −90.2 HU. FAI increases linearly with tube voltage.

---

## 6. Landmark Clinical Trials and Evidence Base

### 6.1 CANTOS (Anti-inflammatory therapy — causal proof)

> Ridker PM et al. *NEJM* 2017. n=10,061. MACE −15% with canakinumab (IL-1β antibody).

Proves inflammation is a causal driver of MACE, not merely associated. Validates the clinical rationale for measuring coronary inflammation.

### 6.2 CRISP-CT (FAI validation — foundational)

> Oikonomou EK et al. *Lancet* 2018. n=1,872 (Erlangen derivation + Cleveland Clinic validation).

- **RCA-FAI** independently predicted cardiac death at 5-year follow-up: **HR 9.04** (95% CI 2.12–38.6, p=0.003)
- FAI added incremental prognostic value beyond CACS, Gensini score, and Framingham Risk Score
- **FAI cut-off: −70.1 HU** identified by ROC analysis (AUC = 0.76 for cardiac death)
- Reproducibility: ICC = 0.987 intraobserver, 0.980 interobserver
- Defined the exact technical parameters (−190 to −30 HU fat window, proximal 40 mm VOI, 1× vessel diameter radial extent)

### 6.3 ORFAN (AI-enhanced FAI — prospective validation)

> Oikonomou EK et al. *Nature Cardiovascular Research* 2023. n=3,324.

- CaRi-Heart AI risk score outperformed ASCVD PCE score, SRS, and CACS alone
- Added incremental value in intermediate-risk patients (10-year ASCVD 7.5–20%)
- Integrates FAI + shape features + coronary calcification into a single AI risk score

**Extended ORFAN results (Chan K, Wahome SKW, Antoniades C et al. 2024, *Lancet*)**:

> "Inflammatory risk and cardiovascular events in patients without obstructive coronary artery disease: the ORFAN multicentre, longitudinal cohort study." *Lancet* 2024.

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

### 6.4 COLCOT and LoDoCo2 (Colchicine)

> Tardif JC et al. *NEJM* 2019 (COLCOT): −23% MACE post-MI.
> Nidorf SM et al. *NEJM* 2020 (LoDoCo2): −31% MACE in chronic CAD.

Established colchicine as an evidence-based anti-inflammatory agent for CAD, creating the treatment pathway that FAI-guided stratification enables.

### 6.5 LoDoCo2 CT Substudy — Key Negative Finding

> Fiolet ATL et al. "Effect of Low-Dose Colchicine on Pericoronary Inflammation and Coronary Plaque Composition in Chronic Coronary Disease: A Sub-Analysis of the LoDoCo2 Trial." *Heart* 2025. n=151 from LoDoCo2 trial (n=5,522 main trial).

Pre-specified cross-sectional substudy. 128-slice Siemens Definition Flash at 4 Dutch sites. Median **28.2 months** of colchicine 0.5 mg daily vs placebo. Used AutoPlaque v3.0 (Cedars-Sinai) for plaque analysis.

| Measurement | Colchicine | Placebo | p-value |
|---|---|---|---|
| **PCAT attenuation** | −79.5 HU | −78.7 HU | **0.236** (NS) |
| hs-CRP | No difference | — | NS |
| IL-6 | No difference | — | NS |
| Calcified plaque volume | **169.6 mm³** | 113.1 mm³ | **0.041** |
| Dense calcified plaque | **192.8 mm³** | 144.3 mm³ | **0.048** |
| LAP burden (low-intensity statin subgroup) | Lower | Higher | p_interaction=**0.037** |

- No correlation between hs-CRP/IL-6 and PCAT attenuation in either group
- Higher calcified plaque in the colchicine arm is consistent with **plaque stabilisation** (calcification = healing)
- **42% had stented proximal segments** — may have affected PCAT measurement via partial volume artifact
- Cross-sectional design (no baseline CT) is a key limitation — cannot assess within-patient change

**Critical implication**: Despite colchicine reducing MACE by 31% in the main LoDoCo2 trial, PCAT attenuation **failed to detect** its anti-inflammatory effect. If FAI cannot capture the signal of a proven anti-inflammatory drug, its utility as a **treatment monitoring biomarker** is questionable. This directly supports the argument that HU-based FAI has fundamental sensitivity limitations.

### 6.6 Meta-Analyses

> Sagris M et al. 2022. 20 studies, n=7,797. FAI significantly higher around unstable vs stable plaques.

Confirms the FAI signal is robust and reproducible across diverse study populations and scanner platforms for distinguishing plaque stability.

---

## 7. Recent FAI Prognostic Evidence (2023–2026)

### 7.1 Risk Reclassification

**Coerkamp et al. (2024/2025, *Int J Cardiol Cardiovasc Risk Prev*, Amsterdam, n=50)**: FAI reclassified **62% of patients** — 22% upgraded to higher risk, 40% downgraded. Used CaRi-Heart commercially. Demonstrates that FAI meaningfully changes clinical decisions in a majority of patients.

### 7.2 Special Populations

#### Diabetes
- **Zhang et al. (2025, *Front Endocrinol*, n=468)**: Longer diabetes duration independently associated with increased FAI in all three major coronary arteries. Linear dose-response relationship.
- **Feng et al. (2026, *Diabetes Metab Syndr Obes*, n=320)**: FAI mediated the relationship between TyG index and OSA risk in type 2 diabetes.
- **Wang et al. (2026, *BMC Med Imaging*, n=400)**: FAI provided incremental prognostic value in diabetic patients with non-obstructive CAD.
- **Lesion-specific FAI in T2DM (2024, *Cardiovasc Diabetol*)**: FAI for MACE prediction in type 2 diabetes.

#### Chronic Kidney Disease
- **Lu et al. (2025, *BMC Nephrol*, n=444)**: High pFAI independently predicted MACE (HR 1.65–2.72) and cardiovascular mortality (HR 2.12–2.90) across all 3 vessels in CAD + CKD patients. Used ShuKun technology. Median follow-up 4.57 years.

#### Heart Failure
- **Yuasa et al. (2025, *JACC Advances*, n=1,196)**: FAI predicted hospitalization for **HFpEF** (heart failure with preserved ejection fraction) — a novel application beyond traditional atherosclerotic endpoints.

#### Young Patients
- **2025 (*BMC Cardiovasc Disord*)**: FAI for MACE prediction in young people — extending the applicability beyond the typical >50 year age group.

### 7.3 Non-Obstructive CAD / MINOCA

> Diau & Lange (2025, *Curr Cardiol Rep*): Comprehensive review of coronary inflammation in non-obstructive CAD and MINOCA.

Key findings across the literature:
- The ORFAN 40,091-patient study showed more cardiac deaths in the non-obstructive group (81% of the cohort)
- EAT, PCAT, FAI, and AI-Risk algorithms all provide value in this population
- **Tognola et al. (2025)**: PCAT and EAT contribute to coronary inflammation in MINOCA patients — CCTA can detect localised inflammation through attenuation changes
- **Port et al. (2025)**: First reported case of abnormal FAI in hypereosinophilic syndrome causing INOCA

### 7.4 Culprit Lesion Identification in ACS

- **Li et al. (2025, *J Comput Assist Tomogr*, n=120)**: FAI identified culprit lesions in ACS with optimal cutoff of −77 HU. AUC 0.970 when combined with stenosis severity (vs 0.939 for stenosis alone). Non-calcified and mixed plaques showed higher FAI.
- **Yang et al. (2025, *Eur J Radiol Open*, n=230 NSTEMI)**: FAIlesion significantly higher at culprit vs non-culprit sites.
- **Huang et al. (2023)**: Combining FAI with CT-FFR significantly improved culprit lesion identification beyond anatomical assessment alone.

### 7.5 Plaque Vulnerability

- **Luo et al. (2026, *J Thorac Dis*)**: FAI predicts vulnerable plaque characteristics and adverse outcomes.
- **PCAT + OCT combined (2025, *Sci Rep*)**: PCAT attenuation associated with plaque vulnerability using combined CCTA and OCT imaging.
- **Wang et al. (2025, *QIMS*)**: Significant correlations between plaque vulnerability features and multiparametric PCAT indices.
- **Li et al. (2025)**: FAI correlates with changes in plaque components before and after coronary plaque formation, interacting with both volume and composition of newly formed plaques, particularly the necrotic core.

---

## 8. FAI Beyond Atherosclerosis: Emerging Applications

### 8.1 Coronary Vasospasm / Prinzmetal Angina

The **Shimokawa group (Tohoku University)** pioneered this application:

- **Ohyama et al. (2016–2017, *Eur Cardiol*)**: Used 18F-FDG PET/CT to show that coronary perivascular FDG uptake (indicating inflammation) was significantly increased at spastic coronary segments. Coronary PCAT volume was increased at spastic segments in vasospastic angina (VSA) patients, with significant positive correlation between PCAT volume and vasoconstricting response to acetylcholine.
- After 23 months of medical treatment, coronary perivascular FDG uptake decreased — demonstrating FAI dynamics with treatment.
- Established that **local adventitial inflammation**, not systemic inflammation, drives coronary spasm.

### 8.2 Myocarditis and Pericarditis

- **Baritussio et al. (2021, *J Clin Med*, n=?)**: Patients with clinically suspected myocarditis presenting with infarct-like symptoms had significantly elevated FAI values compared to controls. One of the first applications in non-ischaemic inflammatory heart conditions.

### 8.3 Cardiac Transplant — Coronary Artery Vasculopathy (CAV)

- **Moser et al. (2023, *Eur Radiol*)**: Perivascular fat attenuation independently predicted cardiac mortality and need for re-transplantation in heart transplant recipients.
- **Lassandro et al. (2026, pilot study)**: Dynamic FAI progression predicted CAV development and outcomes — suggesting FAI could be valuable for non-invasive surveillance of transplant recipients.

### 8.4 COVID-19

- Multiple studies (2020–2022) examined coronary inflammation post-COVID-19 infection, finding elevated FAI values that correlated with cardiovascular complications. The pandemic accelerated interest in PCAT as a marker of systemic vascular inflammation.

### 8.5 Atrial Fibrillation

- Emerging research explores connections between left atrial PCAT and AF substrate, though this remains early-stage.

### 8.6 FAI Dynamics — Serial Measurements

- **Yoshihara et al. (2025)**: Serial FAI measurements in spontaneous coronary artery dissection showed persistent elevation correlating with disease progression.
- **Shimokawa group**: Demonstrated decreased perivascular inflammation (FDG uptake, correlated with FAI) after 23 months of medical treatment for vasospastic angina.
- **Fiolet et al. (2025)**: Colchicine did NOT change PCAT attenuation after 28 months (see §6.5) — raising questions about FAI sensitivity to treatment effects.

---

## 9. PCAT Radiomics: Beyond Mean HU

### 9.1 Rationale

Mean HU (FAI) is a single summary statistic. The spatial distribution, texture, and heterogeneity of PCAT may contain additional prognostic information about the inflammatory microenvironment.

### 9.2 ShuKun Technology Approach

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

### 9.3 Key Radiomics Studies (2024–2026)

- **Shang et al. (2025, *Cardiovasc Diabetol*, n=777, multicentre)**: PCAT radiomics improved MACE prediction beyond traditional risk scores. Combined model C-index: 0.873 (training), 0.824 (validation). Significant reclassification improvement (NRI: 0.256–0.480).
- **Hou/Liu et al. (2024, *Insights Imaging*, n=180)**: PCAT radiomic signature predicted rapid plaque progression. First-order statistics and higher-order texture features were most predictive.
- **Huang et al. (2025, *Front Cardiovasc Med*)**: Compared lesion-specific vs proximal PCAT radiomics models for MACE — lesion-specific approach showed superior performance.
- **PCAT radiomics for INOCA in NAFLD (2025, *BMC Cardiovasc Disord*)**: Radiomics model (AUC 0.734) outperformed simple PCAT attenuation (AUC 0.674) for diagnosing ischaemia with non-obstructive coronary arteries in NAFLD patients.

### 9.4 Critical Limitation: Radiomic Feature Instability

> Wu et al. (2025, Case Western): **78% of radiomic features change >10%** between different contrast perfusion phases.

This means radiomic models trained on one acquisition timing may not generalise to different timing protocols — the same protocol-dependence problem as FAI, but amplified across 93 features.

---

## 10. Technical Confounders and Limitations of FAI

### 10.1 Tube Voltage / kVp

- **Nie S, Molloi S (2025)**: HU variance of **21.9%** across 80–135 kV for identical tissue composition
- **Ma et al. (2020)**: FAI increases linearly with tube voltage (less negative at higher kV)
- **Etter et al. (2022)**: Required conversion factors relative to 120 kVp: 1.267 (80 kVp), 1.08 (100 kVp), 0.947 (140 kVp)

### 10.2 Reconstruction Kernel and Algorithm

- **Lisi et al. (2024, *Eur Radiol*)**: Up to **33 HU intra-individual variation** (34 HU inter-individual) between reconstruction kernels and iterative reconstruction levels. FAI values decrease with sharper kernels (Bv56: −106±2 HU vs smooth Qr36+QIR4: −87±9 HU). Increasing iterative reconstruction strength causes FAI to increase by up to 12 HU.
- **2025 (*Sci Rep*)**: Additional studies confirming differences in reconstruction algorithms for PCAT attenuation.
- The same patient reconstructed with different kernels can be classified as "inflamed" or "non-inflamed" depending on the reconstruction choice.

### 10.3 Patient Body Habitus

- **Nie S, Molloi S (2025)**: 3.6% HU variance between small, medium, and large patient sizes for identical tissue
- Beam hardening and scatter increase with body size, shifting HU values
- FAI less reliable in obese populations; different baseline values at high BMI

### 10.4 Contrast Timing and Perfusion

- **Wu et al. (2025)**: ~**7 HU swing** in PCAT HU from contrast timing differences; ~**15% PCAT volume change**; **78% of radiomic features change >10%** between perfusion phases
- Bolus timing, injection rate, and cardiac output all affect iodine distribution

### 10.5 Scanner Platform and Detector Type

- Energy-integrating detectors vs photon-counting detectors produce systematically different HU values for the same tissue
- **Tremamunno G et al. (*Acad Radiol* 2025;32(3), DOI: 10.1016/j.acra.2024.11.055)**: Intra-individual FAI differences between PCD-CT and conventional CT confirmed measurements are NOT directly comparable
- PCD-CT values require calibration before the −70.1 HU threshold can be applied

### 10.6 Cardiac Phase

Different cardiac phases (systole vs diastole) produce different apparent PCAT attenuation due to volumetric compression and motion.

### 10.7 Fat Threshold Selection

Multiple studies use inconsistent fat threshold definitions (minimum: −200 to −149 HU; maximum: −45 to −30 HU), preventing direct cross-study comparison. The Oxford standard (−190 to −30 HU) is most common but not universal.

### 10.8 Partial Volume Effects

- **Li et al. (2025)**: PCAT density is highest closest to lumen (within 0.5 mm) and decreases with distance. Measurements within 0.75 mm of lumen are susceptible to partial volume effects from adjacent contrast-enhanced vessel.
- **Hell MM, Achenbach S et al. (*JCCT* 2016;10:52–60)**: Argued that variations in CT density of PCAT are primarily attributed to partial volume effects and image interpolation rather than tissue composition or metabolic activity — a fundamental challenge to the biological interpretation of FAI. PCAT attenuation decreased with increasing distance from the vessel and from proximal to distal segments, consistent with partial volume contamination from the contrast-enhanced lumen.

### 10.9 The −70.1 HU Threshold Problem

The FAI threshold was validated on specific scanner platforms with specific protocols (CRISP-CT: conventional CT, 120 kVp). When applied to:
- Different tube voltages → misclassification
- Different reconstruction algorithms → misclassification
- Different scanner platforms (especially PCD-CT) → misclassification
- Longitudinal monitoring with protocol changes → unreliable trend detection

**Chan & Antoniades (2025, editorial comment: "Pericoronary Adipose Tissue Imaging and the Need for Standardized Measurement of Coronary Inflammation")** have acknowledged this problem and proposed the **"FAI Score"** — adjusting raw PCAT attenuation for technical factors (tube voltage, reconstruction), anatomical factors (vessel size, fat volume), and demographic factors (age, sex, BMI) to produce a standardised score. This is the Oxford group's own proposed solution, but it adds complexity and requires calibration datasets. *Note: this is an editorial comment, not a full original research study.*

### 10.10 The Standardization Challenge

> Němečková et al. (2025, *J Clin Med*): "The Perivascular Fat Attenuation Index: Bridging Inflammation and Cardiovascular Disease Risk" — comprehensive review highlighting the urgent need for standardisation.

The field recognises that FAI cannot be widely adopted as a clinical biomarker without solving the protocol-dependence problem. Current approaches:
1. **Conversion factors** (Etter et al.): kVp-specific correction. Partial solution — doesn't address kernel or scanner effects.
2. **FAI Score** (Antoniades): Multi-factor adjustment. Promising but requires large calibration datasets.
3. **Material decomposition** (our approach): Fundamentally protocol-independent — measures composition, not attenuation.

---

## 11. Negative Studies and Criticisms

### 11.1 Studies Finding No Predictive Value

- **Ma et al. (2021, "Focal pericoronary adipose tissue attenuation is related to plaque presence, plaque type, and stenosis severity in coronary CTA")**: Found no difference in PCAT attenuation between patients with and without CAD.
- **Pandey et al. (2020, "Epicardial fat attenuation, not volume, predicts obstructive coronary artery disease and high risk plaque features in patients with atypical chest pain", *Br J Radiol* 2020;93:20200540)**: Patients with obstructive CAD had even **lower** PCAT attenuation than those without — a counterintuitive finding suggesting the FAI signal is not as straightforward as "higher = more inflamed = more disease."

**Boussoussou et al. (2023, "The effect of patient and imaging characteristics on coronary CT angiography assessed pericoronary adipose tissue attenuation and gradient", *JCCT* 2023, Semmelweis/Cedars-Sinai/MGH)**:

> n=**1,652** patients with **zero calcium score** (low-risk). PCAT range: −123 to −51 HU — a wide range even in low-risk patients.

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

Validated on a different scanner (GE CardioGraphe vs Philips Brilliance): average PCAT attenuation was **15 HU higher** on the different scanner for the same patients. Also validated on moderate-to-severe CAD group: same pattern — plaque associations disappeared after correction.

PCAT gradient showed similar results — associations with plaque vanished after adjustment.

**Implication**: The FAI-plaque relationship reported in many studies may be substantially or entirely confounded by imaging parameters and patient characteristics. This directly supports material decomposition as an alternative — measuring composition rather than HU eliminates these confounders.

### 11.2 Treatment Monitoring Failure

- **Fiolet et al. (2025, LoDoCo2 substudy)**: Despite colchicine reducing MACE by 31%, PCAT attenuation did NOT change after 28 months of colchicine treatment. This is a critical finding — if FAI cannot detect the anti-inflammatory effect of a proven drug, its utility as a treatment monitoring tool is questionable.

### 11.3 The Partial Volume Argument

- **Hell MM, Achenbach S, Schuhbaeck A et al. "CT-based analysis of pericoronary adipose tissue density: Relation to cardiovascular risk factors and epicardial adipose tissue volume." *JCCT* 2016;10(1):52–60. DOI: 10.1016/j.jcct.2015.07.011**: Argued that FAI variations are primarily due to partial volume effects and image interpolation, not genuine tissue composition changes. PCAT attenuation decreased with increasing distance from the vessel and from proximal to distal segments — consistent with partial volume contamination from the contrast-enhanced lumen rather than biological gradients. If true, this challenges the entire biological rationale for FAI.
- Counterargument: Antonopoulos et al. (2017) histological validation directly measured tissue composition changes (lipid droplet size, transcription factors, cytokines) that matched CT-derived FAI — suggesting at least some of the signal is biological.

### 11.4 Systematic Review Heterogeneity

- **Tan et al. (2025) meta-analysis**: Found inconsistent FAI methodologies across studies, high heterogeneity in measurements, and uncertain clinical predictive value for MACE in some subgroups.

### 11.5 JACC Comprehensive Review: "Not Ready for Prime Time"

> Tan N et al. "Pericoronary Adipose Tissue as a Marker of Cardiovascular Risk." *JACC* 2023 (Review Topic of the Week). Baker Heart Institute, Melbourne + Cedars-Sinai.

This authoritative JACC review synthesises the state of the PCAT/FAI field and identifies critical unresolved issues:

**Bidirectional signalling theory**: The review formalises the concept that PCAT communicates with the vessel wall in BOTH directions:
- **Inside-to-outside** (vessel → fat): Inflamed coronary arteries secrete cytokines that inhibit adipocyte maturation → smaller, lipid-poor, water-rich adipocytes → higher HU = FAI signal
- **Outside-to-inside** (fat → vessel): Dysfunctional PCAT (in obesity/metabolic syndrome) secretes pro-inflammatory mediators → accelerates plaque development
- This **bidirectional amplification loop** means FAI reflects both the consequence and the cause of coronary inflammation

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

---

## 12. Photon-Counting CT and Spectral Imaging

### 12.1 PCD-CT for PCAT

PCD-CT (Siemens NAEOTOM Alpha, GE Revolution CT) provides simultaneous multi-energy data:
- VMI at any keV (40–190 keV)
- Material decomposition maps (water, iodine, lipid, calcium)
- Effective atomic number (Z-eff) maps
- Ultra-high resolution mode (0.2 mm pixels)

### 12.2 Key PCD-CT Studies

- **Mergen V, Eberhard M, Alkadhi H et al. "Epicardial Adipose Tissue Attenuation and Fat Attenuation Index: Phantom Study and In Vivo Measurements With Photon-Counting Detector CT." *AJR* 2021**: First systematic assessment of EAT/FAI on PCD-CT. Phantom and in vivo (n=30). VMI at 55–80 keV compared to reference 120 kVp EID scan. Fat attenuation varies significantly with VMI energy level — 70 keV VMI approximates 120 kVp but is not identical.
- **Engel et al. (*J Clin Med* 2026)**: First study applying the −70.1 HU FAI threshold on PCD-CT. FAI ≥ −70.1 HU identified more lipid-rich, non-calcified plaques (vulnerable morphology).
- **Tremamunno G, Vecsey-Nagy M, Kravchenko D et al. "Intra-individual Differences in Pericoronary Fat Attenuation Index Measurements Between Photon-counting and Energy-integrating Detector Computed Tomography." *Acad Radiol* 2025;32(3). DOI: 10.1016/j.acra.2024.11.055**: Intra-individual FAI differences between PCD-CT and conventional EID CT confirmed measurements are NOT directly comparable. However, iterative reconstruction minimises most differences, enabling inter-scanner comparability.
- **Kravchenko D, Tremamunno G, Varga-Szemes A et al. "Intra-individual radiomic analysis of pericoronary adipose tissue: Photon-counting detector vs energy-integrating detector CT angiography." *Int J Cardiol* 2025;420:132749 (MUSC/Schoepf group)**: Extended the Tremamunno FAI work to full radiomic analysis — compared PCAT radiomic feature stability between PCD-CT and EID-CT within the same patients.
- **Kahmann J, Nörenberg D, Ayx I et al. "Interrelation of pericoronary adipose tissue texture and coronary artery disease of the left coronary artery in cardiac photon-counting computed tomography." *Front Cardiovasc Med* 2024;11:1499219 (Mannheim)**: PCAT texture analysis and CAD characterization on PCD-CT. Explored radiomic features of LAD and RCA PCAT on the NAEOTOM Alpha platform.
- **Gao et al. (2025, *Eur J Radiol*)**: PCD-CT UHR mode significantly reduced stent blooming artifacts. Stent-specific FAI was lower in PCD-CT vs simulated conventional CT.
- **2025 (*Front Cardiovasc Med*)**: Development of NEW threshold for pericoronary fat attenuation based on **40 keV VMI** from dual-energy spectral CT — recognizing the old threshold doesn't transfer.

### 12.3 VMI Considerations

VMI at 70 keV closely matches conventional 120 kVp CT in noise characteristics, but fat HU values shift approximately **+5 to +15 HU** due to energy-dependent attenuation differences. The −70.1 HU threshold has not been validated on VMI data. The Zurich group (Alkadhi, Eberhard, Mergen) has systematically evaluated these shifts.

---

## 13. Commercial Landscape

### 13.1 Caristo Diagnostics (CaRi-Heart) — Oxford Spinoff

- **Product**: CaRi-Heart (FAI-based risk) + CaRi-Plaque™ (AI plaque analysis)
- **FDA 510(k) clearance**: CaRi-Plaque™ (March 2025)
- **CPT codes**: AMA assigned Category III codes 0992T and 0993T (2025)
- **Medicare reimbursement**: Finalized across hospital and office settings starting 2026
- **Clinical deployment**: First U.S. hospital (NCH) implementing CaRi-Heart
- **NHS study (2024)**: Demonstrated CaRi-Heart could reduce cardiac deaths by 12% in the UK NHS
- **Key advantage**: Only commercial tool with FAI + AI risk score, backed by CRISP-CT and ORFAN data

### 13.2 Cleerly

- **Focus**: AI-based comprehensive coronary plaque analysis (not specifically FAI-focused)
- **Funding**: $106M round (2024)
- **CPT**: Category I code for AI-QCT advanced plaque analyses
- **Coverage**: Aetna, UnitedHealthcare, Cigna, Humana — 86+ million lives
- **Distinction**: Emphasises plaque characterisation (stenosis, composition, remodelling) over pericoronary inflammation

### 13.3 HeartFlow

- **Product**: HeartFlow Plaque Analysis, FDA 510(k) cleared (2025)
- **CPT**: New Category I code for AI-enabled plaque analysis
- **Funding**: $890.5M total
- **Focus**: CT-FFR + plaque analysis, not specifically PCAT/FAI

### 13.4 ShuKun Technology (China)

- **Product**: Peri-coronary Adipose Tissue Analysis Tool + CoronaryDoc®-FFR
- **Funding**: $296.4M total
- **Focus**: PCAT radiomics (93 features), lesion-specific analysis
- **Market**: Strong presence in Asian markets, used in multiple Chinese multicentre studies

### 13.5 Competitive Summary

| Company | FAI Analysis | Plaque Analysis | AI Risk Score | FDA Cleared | Reimbursement |
|---|---|---|---|---|---|
| **Caristo (CaRi-Heart)** | ✅ Core product | ✅ CaRi-Plaque | ✅ CaRi-Heart score | ✅ (2025) | ✅ Medicare 2026 |
| **Cleerly** | ❌ | ✅ Core product | ✅ | ✅ | ✅ Category I CPT |
| **HeartFlow** | ❌ | ✅ | ✅ (with FFR) | ✅ | ✅ Category I CPT |
| **ShuKun** | ✅ Radiomics | ✅ | Partial | Regional (China) | Regional |

---

## 14. Major Research Groups

### 14.1 Oxford / Antoniades Group (UK)

**Key investigators**: Charalambos Antoniades, Evangelos Oikonomou, Kenneth Chan
**Contributions**: Defined FAI, conducted CRISP-CT and ORFAN trials, commercialised CaRi-Heart (Caristo Diagnostics). The foundational group for PCAT as a clinical biomarker. Currently working on FAI Score (standardised FAI), CaRi-Plaque expansion, and multi-centre implementation.
**Key papers**: Antonopoulos et al. *Sci Transl Med* 2017; Oikonomou et al. *Lancet* 2018; Oikonomou et al. *Nat CV Res* 2023; Chan et al. *Lancet* 2024

### 14.2 Erlangen / Achenbach Group (Germany)

**Key investigators**: Stephan Achenbach, Mohamed Marwan
**Contributions**: CRISP-CT derivation cohort (partner with Oxford), early PCD-CT FAI studies.
**Key papers**: Oikonomou et al. *Lancet* 2018 (co-authors); Engel et al. *J Clin Med* 2026

### 14.3 Cedars-Sinai / Monash (USA/Australia)

**Key investigators**: Damini Dey, Andrew Lin, Daniel Berman, Stephen Nicholls, Dennis Wong
**Contributions**: PCAT radiomics, ML models for MACE prediction, statin effects on PCAT, integration with CT-FFR. Key role in ICONIC study. Partnership with Yonsei University (Korea). SCOT-HEART trial analysis of coronary plaque radiomic phenotypes predicting MI.
**Key papers**: Multiple 2019–2025 on PCAT radiomics and ML

### 14.4 Zurich / Alkadhi Group (Switzerland)

**Key investigators**: Hatem Alkadhi, Katharina Eberhard, André Mergen
**Contributions**: Systematic evaluation of PCAT on PCD-CT (NAEOTOM Alpha), kernel and reconstruction effects on FAI. Established that PCD-CT FAI values are not comparable to conventional CT.
**Key papers**: Eberhard et al. 2022–2025; Mergen et al. 2022–2025; Lisi et al. 2024

### 14.5 Groningen / Vliegenthart Group (Netherlands)

**Key investigators**: Rozemarijn Vliegenthart, Riemer Ma
**Contributions**: Established PCAT reference values per vessel per kV, low-kV imaging effects.
**Key papers**: Ma et al. 2020 (reference values)

### 14.6 Case Western / Rajagopalan Group (USA)

**Key investigators**: Sanjay Rajagopalan, Chris Wu, David Wilson
**Contributions**: Quantified PCAT confounds — contrast perfusion timing, volume changes, radiomic feature instability. Their work is the most damaging to FAI's claims of reliability.
**Key papers**: Wu et al. 2025 (7 HU timing swing, 15% volume change, 78% radiomic features affected)

### 14.7 Sendai / Shimokawa Group (Japan — Tohoku University)

**Key investigators**: Hiroaki Shimokawa, Kensuke Ohyama
**Contributions**: First to link PVAT inflammation to vasospastic angina using 18F-FDG PET. Pioneered non-atherosclerotic coronary inflammation research.
**Key papers**: Ohyama et al. 2016–2017

### 14.8 Korean Groups

**Major centres**: Seoul National University, Asan Medical Center, Samsung Medical Center, Yonsei University
**Contributions**: ICONIC study participation, lesion-specific PCAT analysis, combined CCTA-OCT studies, ethnic-specific PCAT validation, PCAT in diabetic populations. Very active publishing output.

### 14.9 Chinese Groups

**Major centres**: Zhongshan Hospital (Fudan), Beijing Anzhen Hospital, West China Hospital, Guangdong Provincial People's Hospital (Shiqun Chen, Jiyan Chen), Shengjing Hospital/China Medical University (Yang Hou)
**Contributions**: Large multicentre cohort studies, PCAT radiomics (Shang et al. 2025), ShuKun technology integration, PCAT in CKD (Lu et al. 2025), PCAT in diabetes. Chinese groups are **extremely active** — producing a large proportion of recent PCAT publications, particularly on radiomics and special populations.

### 14.10 Japanese Groups (Beyond Shimokawa)

**Major centres**: Kobe University, Okayama University, Kawasaki Medical School
**Contributions**: East Asian validation (n=2,172 across 4 hospitals), PCAT in periprocedural myocardial injury, PCAT in chronic coronary syndrome, post-hoc analyses in T2DM cohorts.

### 14.11 European Groups

- **Amsterdam UMC (Coerkamp, Henriques)**: CaRi-Heart clinical deployment studies, risk reclassification
- **Mannheim (Ayx, Froelich, Nörenberg)**: PCD-CT radiomic texture analysis of PCAT
- **Leiden**: Innovative imaging techniques
- **Politecnico di Milano (Nannini, Redaelli)**: PCAT as predictor of functional severity (CT-FFR correlation)

### 14.12 MolloiLab / UCI (Our Group)

**Key investigators**: Sabee Molloi, Shu Nie
**Contributions**: Material decomposition for coronary plaque (DECT, 2021), water-lipid-protein decomposition for PVAT (2025), current XCAT-based simulation study. Unique positioning at the intersection of material decomposition + PCAT — no other group combines these.
**Key papers**: Ding et al. 2021; Nie S, Molloi S. *Int J Cardiovasc Imaging* 2025;41:1091–1101; current study in preparation

---

## 15. Research Trajectory Timeline

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
| 2021 | Lab: material decomposition for coronary plaque (DECT) | Molloi (UCI) | Ding et al. 2021 |
| 2022 | Meta-analysis: FAI in unstable vs stable plaques (n=7,797) | Sagris et al. | Sagris et al. 2022 |
| 2022 | Phantom study: PCATMA affected by kVp and reconstruction | Etter et al. | Etter et al. 2022 |
| 2022–2024 | PCD-CT FAI studies begin | Zurich, Mannheim | Multiple |
| 2023 | ORFAN: CaRi-Heart AI-FAI outperforms conventional risk (n=3,324) | Oxford | *Nat CV Res* 2023 |
| 2023 | Tan JACC review: PCAT "not ready for prime time" | Baker Heart/Cedars-Sinai | Tan et al., *JACC* 2023 |
| 2023 | Boussoussou: No PCAT-CAD correlation after adjustments | Semmelweis/Cedars-Sinai | Boussoussou et al. 2023 |
| 2024 | ORFAN extended: n=40,091, 7.7 yr follow-up | Oxford/multicentre | Chan et al., *Lancet* 2024 |
| 2024 | Lisi: kernel effects up to 33 HU variation | Zurich | *Eur Radiol* 2024 |
| 2024 | PCD-CT vs EID: FAI not comparable | Tremamunno, Schoepf (MUSC) | *Acad Radiol* 2025;32(3) |
| 2025 | **Lab's previous paper**: water-lipid-protein for PVAT (simulation) | Nie S, Molloi (UCI) | *Int J Cardiovasc Imaging* 2025 |
| 2025 | Wu: perfusion confounds (7 HU swing, 78% radiomic instability) | Case Western | Wu et al. 2025 |
| 2025 | Caristo: FDA clearance CaRi-Plaque, CPT codes, Medicare | Oxford/Caristo | — |
| 2025 | LoDoCo2 CT substudy: colchicine does NOT change FAI | Fiolet et al. | *Heart* 2025 |
| 2025 | PCAT radiomics multicentre (n=777) | Shang et al. (China) | *Cardiovasc Diabetol* 2025 |
| 2025 | FAI Score standardization proposed (editorial comment) | Chan & Antoniades | Oxford |
| 2025 | Coerkamp: FAI reclassifies 62% of patients | Amsterdam | *Int J Cardiol CV Risk Prev* |
| 2026 | **Current study**: XCAT + material decomposition for PCAT | Nie S, Molloi (UCI) | *In preparation* |

---

## 16. Study Design Patterns in PCAT Research

### 16.1 Study Types

| Study Type | Proportion | Examples |
|---|---|---|
| **Retrospective cohort** | ~60% | CRISP-CT, most single-centre PCAT studies |
| **Prospective cohort** | ~15% | ORFAN, some NAEOTOM Alpha studies |
| **Meta-analysis / systematic review** | ~10% | Sagris et al. 2022, Tan et al. 2025 |
| **Phantom / simulation** | ~10% | Etter et al. 2022, Nie & Molloi 2025 |
| **Case-control** | ~5% | ACS vs. stable angina comparisons |

### 16.2 PCAT Measurement Methodology

Nearly all clinical PCAT studies follow the **Oxford/CRISP-CT protocol**:
- Fat HU window: −190 to −30 HU
- VOI: outer vessel wall + 1× mean vessel diameter, proximal 40 mm (LAD/LCX) or 10–50 mm (RCA)
- Primary metric: Mean HU of fat-range voxels = FAI
- Threshold: −70.1 HU (high-risk)

Variations:
- **Lesion-specific PCAT** (ShuKun): VOI around individual plaques, not fixed proximal segments
- **Volumetric PCAT**: Total fat-range voxel volume in cm³ (complementary to FAI)
- **Radiomic PCAT**: 93-feature extraction per VOI (GLCM, GLRLM, GLSZM, NGTDM, GLDM)
- **PCATMA**: No fat threshold — measures all tissue attenuation in the pericoronary VOI

### 16.3 Imaging Protocols

| Protocol Element | Most Common | Variations |
|---|---|---|
| Scanner | 64–320 slice MDCT | PCD-CT (NAEOTOM Alpha), DECT |
| Tube voltage | 120 kVp | 80, 100, 135, 140 kVp; VMI 40–190 keV |
| Reconstruction | FBP or hybrid IR | ADMIRE, SAFIRE, DLIR; soft vs sharp kernels |
| Contrast | Iodinated (300–400 mgI/mL) | Bolus tracking or test bolus |
| ECG gating | Retrospective or prospective | Prospective preferred for dose reduction |
| Phase | Best diastole (60–75% R-R) | Systole in high HR patients |

---

## 17. Patient Selection Methods

### 17.1 Common Inclusion Criteria

| Criterion | Typical Requirement | Rationale |
|---|---|---|
| Indication | Clinically indicated CCTA for suspected/known CAD | Ensure clinical relevance |
| Image quality | Adequate for coronary assessment (motion score ≤ 2) | FAI requires clear vessel-fat boundary |
| Contrast enhancement | Adequate opacification (aortic root >250 HU) | Ensure proper contrast timing |
| ECG gating | Successful gating with evaluable phase | Motion-free reconstruction |
| Age | Typically >18 years, often >40 years | CAD prevalence |

### 17.2 Common Exclusion Criteria

| Criterion | Rationale |
|---|---|
| Prior CABG or coronary stenting in target vessel | Metal artifact contaminates VOI |
| Severe coronary calcification (Agatston >1000 in some studies) | Blooming artifact affects adjacent fat HU |
| Anomalous coronary anatomy | VOI construction assumes normal anatomy |
| Severe motion artifact | Unreliable fat-vessel boundary |
| BMI extremes (>40 or <18 in some studies) | Body habitus affects image quality and HU calibration |
| Active systemic infection or autoimmune disease | Confounds inflammatory signal |
| Recent cardiac surgery (<3 months) | Post-surgical inflammation confounds |

### 17.3 Outcome-Based Cohort Design

For prognostic studies:
- **Follow-up**: Minimum 1 year, typically 3–5 years (CRISP-CT: 5 years median; ORFAN extended: 7.7 years)
- **Endpoints**: Cardiac death (primary in CRISP-CT), MACE (composite of cardiac death, MI, revascularisation)
- **Sample size**: n=500–2,000 for adequate event rates (cardiac death rate ~2–5% at 5 years)
- **Multi-centre**: CRISP-CT used Erlangen + Cleveland Clinic; ORFAN extended to 40,091 across multiple sites

### 17.4 Patient Data in Key Studies

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

---

## 18. Material Decomposition — Our Approach

*Note: This section is intentionally brief. For full technical details, see the lab's publications.*

### 18.1 The Core Argument

All of the confounders described in §10 affect **HU values** (the physical measurement underlying FAI) but do NOT affect the **actual tissue composition** (water, lipid, protein, iodine content). Material decomposition — decomposing each voxel into its constituent materials — is inherently protocol-independent because it measures composition, not attenuation.

### 18.2 Lab's Previous Paper

> Nie S, Molloi S. "Quantification of Water and Lipid Composition of Perivascular Adipose Tissue Using Coronary CT Angiography: A Simulation Study." *Int J Cardiovasc Imaging* 2025;41:1091–1101.

Key results:
- Water fraction RMSE: 0.01–0.64% (sufficient to detect the ~5% clinical threshold)
- HU variance across 80–135 kV: **21.9%** (protocol-dependent)
- HU variance across patient sizes: **3.6%** (protocol-dependent)
- **Material decomposition (water fraction) was protocol-independent**: same composition yielded same water fraction regardless of kV or patient size

### 18.3 Current Study

The current study extends this work by using anatomically realistic XCAT phantoms, simulating pericoronary adipose inflammation as increased water content, and decomposing into 4 materials (water, lipid, collagen, iodine). The key demonstration: FAI (HU) differs across protocols for the same tissue, but material decomposition gives consistent composition regardless of protocol.

### 18.4 Why This Matters

For multi-site trials, longitudinal monitoring, and cross-scanner comparisons, a protocol-independent biomarker is essential. The field is increasingly recognising this need (§10.10), and material decomposition provides a fundamentally different approach than the calibration/correction strategies (conversion factors, FAI Score) currently proposed.

---

## 19. Key References

1. Ross R. *NEJM* 1999;340:115–126 — Atherosclerosis as inflammatory disease
2. Libby P et al. *Circulation* 2021 — Inflammasome pathway and plaque vulnerability
3. Ridker PM et al. *NEJM* 2017 (CANTOS) — IL-1β causal role in MACE, n=10,061
4. Tardif JC et al. *NEJM* 2019 (COLCOT) — Colchicine −23% MACE post-MI
5. Nidorf SM et al. *NEJM* 2020 (LoDoCo2) — Colchicine −31% MACE in chronic CAD
6. Antonopoulos AS et al. *Sci Transl Med* 2017 — FAI histological validation, n=453
7. Oikonomou EK et al. *Lancet* 2018 (CRISP-CT) — FAI definition, −70.1 HU, HR 9.04, n=1,872
8. Oikonomou EK et al. *Nat CV Res* 2023 (ORFAN) — CaRi-Heart AI score, n=3,324
9. Chan K et al. *Lancet* 2024 (ORFAN extended) — n=40,091, 7.7 yr follow-up
10. Fiolet ATL et al. *Heart* 2025 — LoDoCo2 CT substudy: colchicine does not change FAI
11. Ma R et al. 2020 — PCAT reference values: LAD −92.4, LCX −88.4, RCA −90.2 HU (n=493)
12. Sagris M et al. 2022 — Meta-analysis: FAI in unstable vs stable plaques, n=7,797
13. Wu C et al. 2025 — Perfusion confounds: 7 HU swing, 78% radiomic features affected
14. Lisi C et al. *Eur Radiol* 2024 — Kernel/reconstruction effects: up to 33 HU intra-individual variation
15. Etter M et al. 2022 — Phantom kVp study: PCATMA conversion factors
16. Nie S, Molloi S. *Int J Cardiovasc Imaging* 2025;41:1091–1101 — Water-lipid-protein decomposition for PVAT
17. Engel et al. *J Clin Med* 2026 — FAI on PCD-CT, plaque vulnerability
18. Shang J et al. *Cardiovasc Diabetol* 2025 — PCAT radiomics multicentre, n=777
19. Lu et al. *BMC Nephrol* 2025 — FAI in CAD + CKD, n=444
20. Coerkamp et al. *Int J Cardiol CV Risk Prev* 2025 — FAI reclassifies 62% of patients
21. Li et al. *J Comput Assist Tomogr* 2025 — FAI for culprit lesions in ACS, n=120
22. Li et al. *QIMS* 2025 — PCATMA vs FAI diagnostic comparison
23. Zhang et al. *Front Endocrinol* 2025 — FAI and diabetes duration, n=468
24. Yuasa et al. *JACC Advances* 2025 — FAI predicts HFpEF hospitalization, n=1,196
25. Diau & Lange. *Curr Cardiol Rep* 2025 — Review: inflammation in non-obstructive CAD/MINOCA
26. Němečková et al. *J Clin Med* 2025 — Review: FAI bridging inflammation and CV risk
27. Tremamunno G et al. *Acad Radiol* 2025;32(3). DOI: 10.1016/j.acra.2024.11.055 — PCD-CT vs EID FAI intra-individual differences
28. Chan & Antoniades 2025 — FAI Score standardization proposal (**editorial comment**: "Pericoronary Adipose Tissue Imaging and the Need for Standardized Measurement of Coronary Inflammation")
29. Ohyama K et al. 2016–2017 — PVAT inflammation in vasospastic angina (18F-FDG PET)
30. Moser PT et al. *Eur Radiol* 2023 — FAI in cardiac transplant
31. Baritussio A et al. *J Clin Med* 2021 — FAI in myocarditis
32. Boussoussou et al. *JCCT* 2023 — No PCAT-CAD correlation after multivariable adjustment for imaging/patient factors (n=1,652)
33. Huang et al. 2025, PMID 41163958 — ShuKun 93-feature PCAT radiomics for MACE
34. Hou/Liu et al. *Insights Imaging* 2024 — PCAT radiomic signature predicts plaque progression
35. Iacobellis G. *Nat Rev Endocrinol* 2015 — EAT biology and measurement
36. Henrichot E et al. *ATVB* 2005 — PVAT macrophage infiltration and inflammation
37. Ding Y, Molloi S. 2021 — DECT material decomposition for coronary plaque
38. Hell MM, Achenbach S et al. *JCCT* 2016;10(1):52–60. DOI: 10.1016/j.jcct.2015.07.011 — PCAT density: partial volume effects and image interpolation
39. Pandey NN et al. *Br J Radiol* 2020;93:20200540 — Epicardial fat attenuation predicts obstructive CAD (counterintuitive lower PCAT in obstructive CAD)
40. Tan N et al. *JACC* 2023 (Review Topic of the Week) — "Pericoronary Adipose Tissue as a Marker of Cardiovascular Risk" — comprehensive review, "not ready for prime time"
41. Kravchenko D, Tremamunno G et al. *Int J Cardiol* 2025;420:132749 — PCD vs EID PCAT radiomic analysis (intra-individual)
42. Kahmann J, Ayx I et al. *Front Cardiovasc Med* 2024;11:1499219 — PCAT texture and CAD on PCD-CT (Mannheim)
43. Mergen V, Eberhard M, Alkadhi H et al. *AJR* 2021 — EAT/FAI phantom + in vivo on PCD-CT (Zurich)
