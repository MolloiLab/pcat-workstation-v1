# Coronary Artery Inflammation and Pericoronary Adipose Tissue: Comprehensive Field Review

**Project**: PCAT Segmentation Pipeline — MolloiLab  
**Author**: Shu Nie  
**Date**: March 2026  
**Scope**: In-depth review of the PCAT/FAI field — clinical motivations, biological basis, landmark trials, prognostic evidence, research groups, study design patterns, patient selection, technical limitations, emerging applications, PCAT radiomics, photon-counting CT, commercial landscape, and open questions. Synthesised from ~85 papers in the coronary inflammation collection, online literature search (2023–2026), and cross-referencing of major reviews.

---

## Table of Contents

- [Part I: Clinical Motivation](#part-i-clinical-motivation--why-measure-coronary-inflammation)
- [Part II: Biological Basis](#part-ii-biological-basis--what-fai-measures)
- [Part III: Clinical Evidence](#part-iii-clinical-evidence--fai-validation-and-prognostic-value)
- [Part IV: The Measurement Problem — A Unified Framework](#part-iv-the-measurement-problem--a-unified-framework)
- [Part V: Technical Confounders — Quantified Evidence](#part-v-technical-confounders--quantified-evidence)
- [Part VI: Negative Studies and Key Disagreements](#part-vi-negative-studies-and-key-disagreements)
- [Part VII: Methodological DNA — How Groups Design Their Studies](#part-vii-methodological-dna--how-groups-design-their-studies)
- [Part VIII: Emerging Directions](#part-viii-emerging-directions)
- [Part IX: Commercial Landscape](#part-ix-commercial-landscape)
- [Part X: Research Trajectory Timeline](#part-x-research-trajectory-timeline)
- [Part XI: Study Design Patterns and Patient Selection](#part-xi-study-design-patterns-and-patient-selection)
- [Part XII: Material Decomposition — Our Approach](#part-xii-material-decomposition--our-approach)
- [References](#references)

---

## Part I: Clinical Motivation — Why Measure Coronary Inflammation?

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

## Part II: Biological Basis — What FAI Measures

### 2.1 Paradigm Shift: Atherosclerosis as an Inflammatory Disease

#### The Historical View (Lipid-Centric)

Until the 1990s, atherosclerosis was conceptualised primarily as a **lipid storage disorder**: LDL cholesterol accumulates in the arterial intima, forms fatty streaks, and progressively obstructs the lumen. While lipid lowering reduces MACE by ~35%, a large residual cardiovascular risk remains even after optimal statin therapy.

#### The Inflammatory Hypothesis

> Ross R. "Atherosclerosis — an inflammatory disease." *NEJM*. 1999;340:115–126.

Ross established the **"response-to-injury" hypothesis**: the primary trigger is endothelial injury (from oxidised LDL, hemodynamic shear stress, hypertension, smoking), which activates an inflammatory cascade:

1. Endothelial activation → upregulation of adhesion molecules (VCAM-1, ICAM-1, E-selectin)
2. Monocyte recruitment → differentiation into macrophages
3. Macrophages engulf oxidised LDL → foam cells
4. Foam cells secrete pro-inflammatory cytokines (IL-1β, IL-6, TNF-α)
5. Smooth muscle cell migration and proliferation → fibrous cap formation
6. Plaque vulnerability determined by cap thickness vs. inflammatory burden

#### Causal Proof: The CANTOS Trial

> Ridker PM et al. *NEJM* 2017;377:1119–1131. n=10,061.

CANTOS enrolled patients with prior MI and elevated hsCRP (≥2 mg/L). Canakinumab (IL-1β antibody) or placebo was given on top of optimal statin therapy:

- **Primary result**: MACE reduced by **15%** at 150 mg dose (p=0.031)
- Effect was **independent of LDL cholesterol** (LDL did not change)
- Dose-dependent reduction in hsCRP, IL-6
- First proof that targeting inflammation — not lipids — reduces cardiovascular events

#### The Inflammasome Pathway

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

### 2.2 PCAT Anatomy and Paracrine Signalling

#### Anatomy

**Pericoronary adipose tissue (PCAT)** is the adipose tissue immediately surrounding the coronary arteries, between the coronary vessels and the epicardial surface:
- Distinct from **epicardial adipose tissue (EAT)**: the full fat depot within the pericardial sac
- Distinct from **paracardial fat**: fat outside the pericardium
- Lacks a fascial barrier between the fat and the adventitia → direct exchange pathway
- Vascularised by the vasa vasorum, sharing the same microvascular supply as the adventitia

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

The FAI increase reflects a **genuine molecular phenotypic shift** in the fat cells, validated histologically by Antonopoulos et al. (2017, *Science Translational Medicine*, n=453 cardiac surgery patients): perivascular fat sampled adjacent to inflamed coronary segments showed reduced lipid droplet size (p<0.001), reduced PPARγ (−2.3-fold), reduced FABP4 (−1.8-fold), and increased IL-6 (+3.1-fold) and TNF-α (+2.7-fold).

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

#### Vasa Vasorum and Microvascular Inflammation

The **vasa vasorum** — the microvascular network supplying the coronary arterial wall — plays a key role in PCAT inflammation. Studies using micro-CT and histology (Kwon et al. 2015; Ritman & Lerman 2007) have shown:

- Neovascularisation of vasa vasorum (adventitial angiogenesis) precedes and promotes atherosclerotic plaque development
- Inflamed vasa vasorum provide the route for inflammatory cell infiltration into the vessel wall and adjacent PCAT
- PCAT surrounding vessels with dense vasa vasorum networks shows higher HU (more inflammation)

### 2.3 PCAT vs. Epicardial Adipose Tissue (EAT)

| Feature | PCAT | EAT |
|---|---|---|
| **Spatial scope** | ~3–5 mm shell around specific vessel segment | Entire pericardial sac |
| **Measurement unit** | Mean HU (attenuation = FAI) | Total volume (cm³) |
| **Clinical signal** | Per-vessel acute inflammatory state | Whole-heart chronic metabolic risk |
| **Segmentation** | Vessel-specific VOI (centerline-based) | Pericardial sac segmentation |
| **Key paper** | Oikonomou 2018, *Lancet* | Iacobellis, multiple reviews |
| **Commercial tool** | CaRi-Heart (Caristo), ShuKun PCAT | Multiple EAT volume tools |

Correlation between EAT volume and RCA-FAI is moderate at best (r ≈ 0.3–0.4). A patient with high EAT volume but low FAI is metabolically obese but coronary arteries not actively inflamed; low EAT volume but high FAI indicates lean but with focal plaque-driven coronary inflammation. Both are independently associated with MACE.

### 2.4 Spatial Heterogeneity of Coronary Inflammation

#### Why Proximal Segments Matter

The proximal coronary segments (LAD 0–40 mm, LCX 0–40 mm, RCA 10–50 mm) are the clinical focus because:
1. **Most plaques form here**: hemodynamic shear stress at bifurcations and proximal curves
2. **Clinical consequence**: proximal stenoses cause more downstream ischemia
3. **CT resolution**: proximal vessels are larger (3–5 mm diameter) → better SNR for FAI

#### Lesion-Specific PCAT

Rather than fixed proximal segments, measuring PCAT adjacent to each individual plaque provides:
- **Higher specificity**: PCAT around a stable fibrous plaque may be low even in the same vessel as an unstable plaque with high PCAT
- **Better MACE prediction**: the PCAT signal is strongest immediately adjacent to the vulnerable plaque (Huang et al. 2025)

Li et al. (2025, *QIMS*) compared PCATMA (Pericoronary Adipose Tissue Mean Attenuation — not constrained by fat threshold) with FAI and found PCATMA showed significant differences for non-calcified plaque (P<0.001) and mixed plaque (P=0.047), while FAI did not — suggesting threshold-free measurement may be more sensitive.

#### RCA vs. LAD vs. LCX: Different Biology

The clinical literature focuses primarily on RCA-FAI because:
- RCA has the most pericoronary fat (largest fat depot, cleaner VOI)
- LAD runs in the anterior interventricular groove — also well-studied
- LCX runs in the atrioventricular groove adjacent to the left atrial wall → VOI contamination more common

Reference values (Ma et al. 2020, Groningen, n=493): LAD −92.4 HU, LCX −88.4 HU, RCA −90.2 HU. FAI increases linearly with tube voltage.

---

## Part III: Clinical Evidence — FAI Validation and Prognostic Value

### 3.1 Foundational Studies

#### Antonopoulos 2017 (Histological Validation)

> Antonopoulos et al. *Sci Transl Med* 2017. n=453 cardiac surgery patients.

The landmark study that introduced FAI, providing histological validation that PCAT attenuation changes reflect genuine molecular phenotypic shifts in adipocytes: reduced lipid droplet size (p<0.001), reduced PPARγ (−2.3-fold), reduced FABP4 (−1.8-fold), and increased IL-6 (+3.1-fold) and TNF-α (+2.7-fold) in fat adjacent to inflamed coronary segments.

#### CRISP-CT 2018 (FAI Definition and Prognostic Validation)

> Oikonomou EK et al. *Lancet* 2018. n=1,872 (Erlangen derivation + Cleveland Clinic validation).

- **RCA-FAI** independently predicted cardiac death at 5-year follow-up: **HR 9.04** (95% CI 2.12–38.6, p=0.003)
- FAI added incremental prognostic value beyond CACS, Gensini score, and Framingham Risk Score
- **FAI cut-off: −70.1 HU** identified by ROC analysis (AUC = 0.76 for cardiac death)
- Reproducibility: ICC = 0.987 intraobserver, 0.980 interobserver
- Defined the exact technical parameters (−190 to −30 HU fat window, proximal 40 mm VOI, 1× vessel diameter radial extent)

#### ORFAN 2023 & Extended 2024 (AI-Enhanced FAI at Scale)

> Oikonomou EK et al. *Nature Cardiovascular Research* 2023. n=3,324.
> Chan K et al. *Lancet* 2024. n=40,091.

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

> Sagris M et al. 2022. 20 studies, n=7,797.

FAI significantly higher around unstable vs stable plaques, confirming the FAI signal is robust and reproducible across diverse study populations and scanner platforms.

### 3.2 Colchicine Trials and the Treatment Monitoring Question

#### COLCOT and LoDoCo2

> Tardif JC et al. *NEJM* 2019 (COLCOT): −23% MACE post-MI.
> Nidorf SM et al. *NEJM* 2020 (LoDoCo2): −31% MACE in chronic CAD.

Established colchicine as an evidence-based anti-inflammatory agent for CAD, creating the treatment pathway that FAI-guided stratification enables.

#### LoDoCo2 CT Substudy — Key Negative Finding

> Fiolet ATL et al. *Heart* 2025. n=151 from LoDoCo2 trial (n=5,522 main trial).

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

### 3.3 Prognostic Applications (2023–2026)

#### Risk Reclassification

**Coerkamp et al. (2024/2025, *Int J Cardiol Cardiovasc Risk Prev*, Amsterdam, n=50)**: FAI reclassified **62% of patients** — 22% upgraded to higher risk, 40% downgraded. Used CaRi-Heart commercially.

#### Culprit Lesion Identification in ACS

- **Li et al. (2025, *J Comput Assist Tomogr*, n=120)**: FAI identified culprit lesions in ACS with optimal cutoff of −77 HU. AUC 0.970 when combined with stenosis severity (vs 0.939 for stenosis alone). Non-calcified and mixed plaques showed higher FAI.
- **Yang et al. (2025, *Eur J Radiol Open*, n=230 NSTEMI)**: FAIlesion significantly higher at culprit vs non-culprit sites.
- **Huang et al. (2023)**: Combining FAI with CT-FFR significantly improved culprit lesion identification beyond anatomical assessment alone.

#### Plaque Vulnerability

- **Luo et al. (2026, *J Thorac Dis*)**: FAI predicts vulnerable plaque characteristics and adverse outcomes.
- **PCAT + OCT combined (2025, *Sci Rep*)**: PCAT attenuation associated with plaque vulnerability using combined CCTA and OCT imaging.
- **Wang et al. (2025, *QIMS*)**: Significant correlations between plaque vulnerability features and multiparametric PCAT indices.
- **Li et al. (2025)**: FAI correlates with changes in plaque components before and after coronary plaque formation, interacting with both volume and composition of newly formed plaques, particularly the necrotic core.

#### Special Populations

**Diabetes**:
- **Zhang et al. (2025, *Front Endocrinol*, n=468)**: Longer diabetes duration independently associated with increased FAI in all three major coronary arteries. Linear dose-response relationship.
- **Feng et al. (2026, *Diabetes Metab Syndr Obes*, n=320)**: FAI mediated the relationship between TyG index and OSA risk in type 2 diabetes.
- **Wang et al. (2026, *BMC Med Imaging*, n=400)**: FAI provided incremental prognostic value in diabetic patients with non-obstructive CAD.
- **Lesion-specific FAI in T2DM (2024, *Cardiovasc Diabetol*)**: FAI for MACE prediction in type 2 diabetes.

**Chronic Kidney Disease**:
- **Lu et al. (2025, *BMC Nephrol*, n=444)**: High pFAI independently predicted MACE (HR 1.65–2.72) and cardiovascular mortality (HR 2.12–2.90) across all 3 vessels in CAD + CKD patients. Used ShuKun technology. Median follow-up 4.57 years.

**Heart Failure**:
- **Yuasa et al. (2025, *JACC Advances*, n=1,196)**: FAI predicted hospitalization for **HFpEF** (heart failure with preserved ejection fraction) — a novel application beyond traditional atherosclerotic endpoints.

**Young Patients**:
- **2025 (*BMC Cardiovasc Disord*)**: FAI for MACE prediction in young people — extending the applicability beyond the typical >50 year age group.

#### Non-Obstructive CAD / MINOCA

> Diau & Lange (2025, *Curr Cardiol Rep*): Comprehensive review of coronary inflammation in non-obstructive CAD and MINOCA.

- The ORFAN 40,091-patient study showed more cardiac deaths in the non-obstructive group (81% of the cohort)
- **Tognola et al. (2025)**: PCAT and EAT contribute to coronary inflammation in MINOCA patients
- **Port et al. (2025)**: First reported case of abnormal FAI in hypereosinophilic syndrome causing INOCA

### 3.4 FAI Beyond Atherosclerosis

#### Coronary Vasospasm / Prinzmetal Angina

The **Shimokawa group (Tohoku University)** pioneered this application:
- **Ohyama et al. (2016–2017, *Eur Cardiol*)**: Used 18F-FDG PET/CT to show that coronary perivascular FDG uptake was significantly increased at spastic coronary segments. PCAT volume was increased at spastic segments in vasospastic angina patients, with significant positive correlation between PCAT volume and vasoconstricting response to acetylcholine.
- After 23 months of medical treatment, coronary perivascular FDG uptake decreased — demonstrating FAI dynamics with treatment.
- Established that **local adventitial inflammation**, not systemic inflammation, drives coronary spasm.

#### Myocarditis and Pericarditis

- **Baritussio et al. (2021, *J Clin Med*)**: Patients with clinically suspected myocarditis presenting with infarct-like symptoms had significantly elevated FAI values compared to controls. One of the first applications in non-ischaemic inflammatory heart conditions.

#### Cardiac Transplant — Coronary Artery Vasculopathy (CAV)

- **Moser et al. (2023, *Eur Radiol*)**: Perivascular fat attenuation independently predicted cardiac mortality and need for re-transplantation in heart transplant recipients.
- **Lassandro et al. (2026, pilot study)**: Dynamic FAI progression predicted CAV development and outcomes.

#### COVID-19

- Multiple studies (2020–2022) examined coronary inflammation post-COVID-19 infection, finding elevated FAI values that correlated with cardiovascular complications. The pandemic accelerated interest in PCAT as a marker of systemic vascular inflammation.

#### Atrial Fibrillation

- Emerging research explores connections between left atrial PCAT and AF substrate, though this remains early-stage.

#### Serial Measurements & Treatment Monitoring

- **Yoshihara et al. (2025)**: Serial FAI measurements in spontaneous coronary artery dissection showed persistent elevation correlating with disease progression.
- **Shimokawa group**: Demonstrated decreased perivascular inflammation after 23 months of medical treatment for vasospastic angina.
- **Fiolet et al. (2025)**: Colchicine did NOT change PCAT attenuation after 28 months (see §3.2) — raising questions about FAI sensitivity to treatment effects.

---

## Part IV: The Measurement Problem — A Unified Framework

### 4.1 The Core Inverse Problem

All PCAT measurement approaches attempt to solve the same inverse problem. What the CT scanner measures is:

```
HU_measured = f(tissue_composition) + g(imaging_parameters) + noise
```

Where:
- **f(tissue_composition)** = the biological signal of interest (water content, lipid content, inflammation state)
- **g(imaging_parameters)** = systematic bias from kVp, reconstruction kernel, scanner model, contrast timing, patient size, cardiac phase
- **noise** = random measurement error (photon statistics, electronic noise)

The clinical goal is to extract **f** — but every FAI measurement contains an inseparable mixture of **f + g**. All approaches in the literature are strategies for **deconvolving f from g**. They differ only in how they handle g.

### 4.2 Taxonomy of Deconvolution Strategies

| Strategy | Approach | Representative Work | How g Is Handled | Assumption | Limitation |
|---|---|---|---|---|---|
| **Ignore confounders** | Raw FAI (mean HU in fat window) | Oxford (CRISP-CT, ORFAN) | Assume g is negligible relative to f | Bio variance >> tech variance | Fails across protocols; 21.9% HU variance from kVp alone |
| **Single-parameter correction** | kVp conversion factors | Etter et al. (Zurich) 2022 | Correct for the largest single confounder | Only kVp matters | Other confounders (kernel, scanner, contrast) remain |
| **Multi-factor correction** | FAI Score (standardised FAI) | Antoniades (Oxford) 2025 | Regression-based adjustment for kVp, kernel, anatomy, demographics | g capturable by regression model | Residual confounding; requires large calibration datasets |
| **Feature enrichment** | Radiomics (93+ features) | ShuKun, Cedars-Sinai | Extract more information from the same HU data | Additional signal hidden in texture | g contaminates ALL features — 78% unstable across protocols |
| **Domain expansion** | PCATMA (no fat threshold) | Li et al. 2025 | Sample more completely within VOI | More complete sampling captures more biology | Still HU-based; all g confounders remain |
| **Paradigm shift** | Material decomposition | Nie & Molloi (UCI) 2025 | Eliminate g entirely — measure composition, not attenuation | Physics-based material separation | Requires multi-energy CT (DECT/PCD-CT); not yet clinically validated |

### 4.3 The Trade-Off Space

Each strategy makes a trade-off between **protocol dependence** and **measurement complexity**:

```
Protocol-specific                                      Protocol-independent
(simple, widely available)                              (complex, requires special hardware)

Raw FAI ← kVp correction ← FAI Score ← Radiomics ← PCATMA ← Material Decomposition
  ↑            ↑               ↑           ↑            ↑            ↑
Ignores g   Fixes 1        Fixes many   More features  More voxels  Eliminates g
            confounder     confounders  (all confounded) (all confounded) entirely
```

The critical insight: **all strategies left of material decomposition remain in the HU domain**, meaning they are fundamentally limited by the fact that HU encodes both biology and physics inseparably. Each strategy is a more sophisticated attempt to untangle this mixture — but the mixture persists.

Material decomposition sidesteps the problem entirely by operating in the **composition domain** (water fraction, lipid fraction), where imaging parameters have no systematic effect.

### 4.4 What This Framework Reveals

1. **The field's disagreements are not about data — they are about assumptions.** Oxford assumes bio variance >> tech variance. Erlangen and Boussoussou showed tech variance ≈ bio variance. Both are right about their data; they differ about what the data means.

2. **There is no "best" correction strategy within the HU domain.** Each correction removes some confounders but introduces new assumptions. The only way to eliminate the problem is to leave the HU domain entirely.

3. **The question "Does FAI work?" is ill-posed.** The real question is: "Under what protocol conditions does the biological signal exceed the technical noise?" The answer defines each approach's valid operating range.

---

## Part V: Technical Confounders — Quantified Evidence

### 5.1 Tube Voltage / kVp

- **Nie S, Molloi S (2025)**: HU variance of **21.9%** across 80–135 kV for identical tissue composition
- **Ma et al. (2020)**: FAI increases linearly with tube voltage (less negative at higher kV)
- **Etter et al. (2022)**: Required conversion factors relative to 120 kVp: 1.267 (80 kVp), 1.08 (100 kVp), 0.947 (140 kVp)

### 5.2 Reconstruction Kernel and Algorithm

- **Lisi et al. (2024, *Eur Radiol*)**: Up to **33 HU intra-individual variation** (34 HU inter-individual) between reconstruction kernels and iterative reconstruction levels. FAI values decrease with sharper kernels (Bv56: −106±2 HU vs smooth Qr36+QIR4: −87±9 HU). Increasing iterative reconstruction strength causes FAI to increase by up to 12 HU.
- **2025 (*Sci Rep*)**: Additional studies confirming differences in reconstruction algorithms for PCAT attenuation.
- The same patient reconstructed with different kernels can be classified as "inflamed" or "non-inflamed" depending on the reconstruction choice.

### 5.3 Patient Body Habitus

- **Nie S, Molloi S (2025)**: 3.6% HU variance between small, medium, and large patient sizes for identical tissue
- Beam hardening and scatter increase with body size, shifting HU values
- FAI less reliable in obese populations; different baseline values at high BMI

### 5.4 Contrast Timing and Perfusion

- **Wu et al. (2025)**: ~**7 HU swing** in PCAT HU from contrast timing differences; ~**15% PCAT volume change**; **78% of radiomic features change >10%** between perfusion phases
- Bolus timing, injection rate, and cardiac output all affect iodine distribution

### 5.5 Scanner Platform and Detector Type

- Energy-integrating detectors vs photon-counting detectors produce systematically different HU values for the same tissue
- **Tremamunno G et al. (*Acad Radiol* 2025;32(3))**: Intra-individual FAI differences between PCD-CT and conventional CT confirmed measurements are NOT directly comparable
- PCD-CT values require calibration before the −70.1 HU threshold can be applied

### 5.6 Cardiac Phase

Different cardiac phases (systole vs diastole) produce different apparent PCAT attenuation due to volumetric compression and motion.

### 5.7 Fat Threshold Selection

Multiple studies use inconsistent fat threshold definitions (minimum: −200 to −149 HU; maximum: −45 to −30 HU), preventing direct cross-study comparison. The Oxford standard (−190 to −30 HU) is most common but not universal.

### 5.8 Partial Volume Effects

- **Li et al. (2025)**: PCAT density is highest closest to lumen (within 0.5 mm) and decreases with distance. Measurements within 0.75 mm of lumen are susceptible to partial volume effects from adjacent contrast-enhanced vessel.
- **Hell MM, Achenbach S et al. (*JCCT* 2016;10:52–60)**: Argued that variations in CT density of PCAT are primarily attributed to partial volume effects and image interpolation rather than tissue composition or metabolic activity — a fundamental challenge to the biological interpretation of FAI. PCAT attenuation decreased with increasing distance from the vessel and from proximal to distal segments, consistent with partial volume contamination from the contrast-enhanced lumen.

### 5.9 The −70.1 HU Threshold Problem

The FAI threshold was validated on specific scanner platforms with specific protocols (CRISP-CT: conventional CT, 120 kVp). When applied to:
- Different tube voltages → misclassification
- Different reconstruction algorithms → misclassification
- Different scanner platforms (especially PCD-CT) → misclassification
- Longitudinal monitoring with protocol changes → unreliable trend detection

**Chan & Antoniades (2025, editorial comment)** have acknowledged this problem and proposed the **"FAI Score"** — adjusting raw PCAT attenuation for technical factors (tube voltage, reconstruction), anatomical factors (vessel size, fat volume), and demographic factors (age, sex, BMI) to produce a standardised score. *Note: this is an editorial comment, not a full original research study.*

### 5.10 The Standardisation Challenge

> Němečková et al. (2025, *J Clin Med*): "The Perivascular Fat Attenuation Index: Bridging Inflammation and Cardiovascular Disease Risk" — comprehensive review highlighting the urgent need for standardisation.

The field recognises that FAI cannot be widely adopted as a clinical biomarker without solving the protocol-dependence problem. Current approaches:
1. **Conversion factors** (Etter et al.): kVp-specific correction. Partial solution — doesn't address kernel or scanner effects.
2. **FAI Score** (Antoniades): Multi-factor adjustment. Promising but requires large calibration datasets.
3. **Material decomposition** (our approach): Fundamentally protocol-independent — measures composition, not attenuation.

---

## Part VI: Negative Studies and Key Disagreements

### 6.1 Studies Finding No Predictive Value

- **Ma et al. (2021)**: Found no difference in PCAT attenuation between patients with and without CAD.
- **Pandey et al. (2020, *Br J Radiol* 2020;93:20200540)**: Patients with obstructive CAD had even **lower** PCAT attenuation than those without — a counterintuitive finding suggesting the FAI signal is not as straightforward as "higher = more inflamed = more disease."

### 6.2 Boussoussou 2023: The Most Damaging Evidence

> Boussoussou et al. *JCCT* 2023. n=1,652 patients with zero calcium score (low-risk). PCAT range: −123 to −51 HU.

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

**Implication**: The FAI-plaque relationship reported in many studies may be substantially or entirely confounded by imaging parameters and patient characteristics.

### 6.3 Treatment Monitoring Failure

- **Fiolet et al. (2025, LoDoCo2 substudy)**: Despite colchicine reducing MACE by 31%, PCAT attenuation did NOT change after 28 months of colchicine treatment (see §3.2 for full data table).

### 6.4 Systematic Review Heterogeneity

- **Tan et al. (2025) meta-analysis**: Found inconsistent FAI methodologies across studies, high heterogeneity in measurements, and uncertain clinical predictive value for MACE in some subgroups.

### 6.5 JACC Comprehensive Review: "Not Ready for Prime Time"

> Tan N et al. "Pericoronary Adipose Tissue as a Marker of Cardiovascular Risk." *JACC* 2023 (Review Topic of the Week). Baker Heart Institute, Melbourne + Cedars-Sinai.

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

### 6.6 Reinterpreting the Disagreements: Four Paradoxes

Rather than treating the positive and negative studies as contradictions, they can be understood as four manifestations of the same underlying measurement problem (Part IV):

#### Paradox 1: The Oxford-Erlangen Paradox

Oxford and Erlangen **co-authored CRISP-CT** (2018) — the foundational study validating FAI. Yet Erlangen (Hell et al. 2016) published the most cited partial volume critique, arguing FAI variations are primarily imaging artifact. How can the same group both validate and undermine the biomarker?

**Resolution**: They are not contradicting each other — they are addressing different aspects of the inverse problem. Oxford's CRISP-CT shows that **f + g predicts outcomes** (true: even confounded measurements can be prognostic if confounders correlate with disease). Erlangen's critique shows that **g is large relative to f** (also true: partial volume effects are substantial). Both can be right simultaneously — the question is whether f alone (biology) would still predict outcomes if g were removed.

#### Paradox 2: The Correction Paradox

As statistical correction for confounders becomes more rigorous, the biological signal **disappears**. Boussoussou (2023) showed that after correcting for kVp, pixel spacing, BMI, heart rate, and CNR, PCAT attenuation had **zero association with plaque** (p=0.93).

**Resolution**: This does not necessarily mean FAI has no biological content. It could mean: (a) the confounders are so large they overwhelm the signal, (b) the correction over-adjusts by removing variance that is both biological and technical, or (c) the signal truly is artifactual. The correct interpretation depends on whether material decomposition — which eliminates g without regression — still shows a biological signal. This is exactly what our approach tests.

#### Paradox 3: The Clinical Paradox

ORFAN shows FAI Score predicts cardiac mortality with HR 29.8 (3 inflamed arteries vs none). But LoDoCo2 CT substudy shows colchicine (a proven anti-inflammatory that reduces MACE by 31%) does NOT change FAI.

**Resolution**: FAI may function as a **risk marker** (correlates with outcomes) without being a **treatment target marker** (changes with effective therapy). This distinction is critical: a measurement can predict who will have events without being sensitive enough to detect the mechanism by which treatment prevents them. Alternatively, colchicine may work through a pathway (neutrophil-mediated, inflammasome-downstream) that does not change adipocyte phenotype — the biological substrate FAI measures.

#### Paradox 4: The Radiomics Paradox

More features should capture more biology than a single summary statistic (mean HU). Yet 78% of radiomic features are unstable across perfusion phases (Wu et al. 2025), and radiomics models have not consistently outperformed simple FAI.

**Resolution**: Radiomics enriches the **signal space** (extracting texture, heterogeneity, etc.) but equally enriches the **noise space** — each feature is independently contaminated by g. More features × same confounders = more opportunities for overfitting to protocol-specific patterns. This explains why radiomic models often fail external validation despite strong internal performance.

---

## Part VII: Methodological DNA — How Groups Design Their Studies

### 7.1 Overview

Each research group brings a distinct set of assumptions, validated through specific design choices, that creates characteristic strengths and blind spots. Understanding this "methodological DNA" reveals why groups reach different conclusions from similar data.

### 7.2 Oxford / Antoniades Group (UK)

**Core Assumption**: *"Build the largest evidence base; the signal must be real because outcomes are real."*

**Key investigators**: Charalambos Antoniades, Evangelos Oikonomou, Kenneth Chan

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| Multicentre retrospective cohorts (CRISP-CT: Erlangen+Cleveland; ORFAN: 8 NHS hospitals) | Maximise sample size and generalisability | No scanner standardisation — different platforms contribute uncontrolled variance to g |
| Proprietary software (CaRi-Heart) | Integrate FAI + AI + plaque into commercial product | Black-box algorithm limits independent reproducibility |
| Progressive complexity (FAI → FAI Score → AI-Risk) | Each iteration addresses criticisms of the last | Each iteration adds parameters, reducing interpretability |
| Outcome-driven validation (cardiac death, MACE) | Most clinically relevant endpoint | Does not prove biological mechanism — confounded measurement can still predict outcomes if confounders correlate with disease |

**DNA Summary**: Oxford approaches FAI as a **clinical tool** first. If it predicts outcomes, it works — the mechanism is secondary. This is pragmatically powerful (FDA clearance, Medicare reimbursement) but leaves the biological question unresolved.

**Key papers**: Antonopoulos *Sci Transl Med* 2017; Oikonomou *Lancet* 2018; Oikonomou *Nat CV Res* 2023; Chan *Lancet* 2024

### 7.3 Erlangen / Achenbach Group (Germany)

**Core Assumption**: *"Burden of proof is on biology, not physics."*

**Key investigators**: Stephan Achenbach, Mohamed Marwan, Michaela M. Hell

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| Distance-from-lumen analysis (Hell 2016) | Test whether FAI gradient follows physics (partial volume) or biology (inflammation diffusion) | Cannot distinguish partial volume from genuine biological gradient that happens to decrease with distance |
| Co-authorship with Oxford on CRISP-CT | Contribute derivation cohort while maintaining independent critique | Creates appearance of inconsistency when same group both validates and critiques |
| Early PCD-CT FAI evaluation (Engel 2026) | Assess FAI on next-generation hardware | Small sample sizes limit generalisability |

**DNA Summary**: Erlangen applies **physics-first skepticism** — any biological claim must first survive the null hypothesis that the signal is artifactual. This makes them the field's most rigorous internal critics, but risks dismissing real biological signals that happen to co-occur with physical confounders.

**Key papers**: Hell *JCCT* 2016; Oikonomou *Lancet* 2018 (co-authors); Engel *J Clin Med* 2026

### 7.4 Cedars-Sinai / Monash (USA/Australia)

**Core Assumption**: *"If the signal disappears under scrutiny, question whether it existed."*

**Key investigators**: Damini Dey, Andrew Lin, Daniel Berman, Stephen Nicholls, Dennis Wong

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| ICONIC prospective registry | Controlled acquisition protocol | Single-centre protocol may not reflect real-world variation |
| 1,103 radiomic parameters (maximum feature extraction) | Cast widest net for prognostic features | More features = more multiple comparison problems |
| Random Forest ML with strict protocol matching | Reduce overfitting risk | Protocol matching limits generalisability to that specific protocol |
| Partnership with Boussoussou (Semmelweis) | Independent confounder analysis | Zero-calcium-score cohort may not represent typical FAI population |

**DNA Summary**: Cedars-Sinai brings **methodological rigour from radiomics** but inherits the fundamental limitation: all features are extracted from confounded HU data. Their most impactful contribution may be the negative finding (Boussoussou 2023) rather than the positive radiomics models.

**Key papers**: Multiple 2019–2025 on PCAT radiomics; Boussoussou *JCCT* 2023

### 7.5 Zurich / Alkadhi Group (Switzerland)

**Core Assumption**: *"Characterise the instrument before measuring biology."*

**Key investigators**: Hatem Alkadhi, Katharina Eberhard, André Mergen

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| Same raw data reconstructed with multiple algorithms | Isolate reconstruction effect (within-patient, within-scan) | Reconstruction is only one of many confounders |
| Repeated measures ANOVA (within-subject design) | Gold standard for isolating single-variable effects | Doesn't address interactions between confounders |
| Phantom validation before in vivo | Establish ground truth where tissue composition is known | Phantoms don't fully replicate in vivo complexity |
| Systematic PCD-CT evaluation (NAEOTOM Alpha) | First-mover on next-generation hardware | Limited to single PCD-CT platform |

**DNA Summary**: Zurich applies **metrological rigour** — systematically quantifying each source of measurement error. This is essential foundational work but does not propose a solution, only characterises the problem.

**Key papers**: Eberhard 2022–2025; Mergen *AJR* 2021; Lisi *Eur Radiol* 2024

### 7.6 Case Western / Rajagopalan Group (USA)

**Core Assumption**: *"If the foundation is unstable, the building cannot stand."*

**Key investigators**: Sanjay Rajagopalan, Chris Wu, David Wilson

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| Dynamic CT perfusion (multi-phase acquisition) | Capture temporal variation in PCAT measurements | Perfusion CT is not standard CCTA — confounders may differ |
| Temporal analysis (contrast timing effects) | Show that the same tissue gives different FAI at different time points | Single-centre, moderate sample size (n=135) |
| Full radiomic feature stability assessment | Quantify which features survive protocol variation | Does not assess which features would be stable under material decomposition |

**DNA Summary**: Case Western provides the **most damaging quantitative evidence** against FAI reliability (7 HU swing, 15% volume change, 78% radiomic instability). Their work establishes the floor of measurement uncertainty that any approach must exceed to claim biological sensitivity.

**Key papers**: Wu et al. 2025

### 7.7 Groningen / Vliegenthart Group (Netherlands)

**Core Assumption**: *"If you can't eliminate protocol effects, quantify them."*

**Key investigators**: Rozemarijn Vliegenthart, Riemer Ma

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| Consecutive CCTA cohort (n=493) with no known CAD | Establish clean reference values in healthy population | "No known CAD" ≠ no subclinical inflammation |
| Per-vessel, per-kV reference ranges | Provide platform-specific reference data | Reference values are still protocol-specific — not a solution, just calibration |
| Linear regression: FAI vs kV | Simple, interpretable model | Assumes linearity; doesn't account for kernel × kV interactions |

**DNA Summary**: Groningen provides essential **normative data** but does not propose a mechanism or solution — their work defines the problem's magnitude.

**Key papers**: Ma et al. 2020

### 7.8 Tohoku / Shimokawa Group (Japan)

**Core Assumption**: *"If FAI works in atherosclerosis, it should work in other inflammatory states."*

**Key investigators**: Hiroaki Shimokawa, Kensuke Ohyama

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| 18F-FDG PET/CT (not just CCTA) | Gold standard for metabolic inflammation imaging | PET has its own limitations (resolution, uptake confounds) |
| Vasospastic angina focus | Extend FAI beyond atherosclerosis | Small sample sizes; different pathophysiology may not generalise |
| Serial measurements (pre/post-treatment) | Test FAI dynamics | PET-based measurement, not CT-based FAI — results may not transfer |

**DNA Summary**: Shimokawa established PCAT as a **general inflammatory biomarker**, not just an atherosclerosis tool. Their PET-validated approach provides the strongest mechanistic evidence but using a different modality than the CT-based FAI used by other groups.

**Key papers**: Ohyama et al. 2016–2017

### 7.9 Chinese Groups / ShuKun Technology

**Core Assumption**: *"Scale + features + AI → patterns emerge."*

**Major centres**: Zhongshan Hospital (Fudan), Beijing Anzhen Hospital, West China Hospital, Guangdong Provincial People's Hospital (Shiqun Chen, Jiyan Chen), Shengjing Hospital/China Medical University (Yang Hou)

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| Large multicentre cohorts (Shang 2025: n=777, 3 centres) | Statistical power for subgroup analyses | Heterogeneous protocols across centres |
| Lesion-specific VOI (ShuKun pipeline) | Higher anatomical specificity than fixed proximal segments | VOI placement requires plaque identification — circular if used to predict plaque |
| 1,103 radiomic features with binning | Comprehensive feature extraction | Feature instability across protocols (see Paradox 4) |
| Custom R pipeline (Pearson → Lasso → XGBoost) | Standard ML workflow | Black-box risk; limited mechanistic insight |

**DNA Summary**: Chinese groups produce the **highest publication volume** in PCAT research, particularly on radiomics and special populations (diabetes, CKD). Their work establishes PCAT's prognostic value in East Asian populations but inherits all HU-domain limitations.

**Key papers**: Shang *Cardiovasc Diabetol* 2025; Lu *BMC Nephrol* 2025; Huang *Front Cardiovasc Med* 2025

### 7.10 Korean Groups

**Major centres**: Seoul National University, Asan Medical Center, Samsung Medical Center, Yonsei University

Contributions: ICONIC study participation, lesion-specific PCAT analysis, combined CCTA-OCT studies, ethnic-specific PCAT validation, PCAT in diabetic populations. Very active publishing output with emphasis on multimodality imaging (CCTA + OCT, CCTA + CT-FFR).

### 7.11 Japanese Groups (Beyond Shimokawa)

**Major centres**: Kobe University, Okayama University, Kawasaki Medical School

Contributions: East Asian validation (n=2,172 across 4 hospitals), PCAT in periprocedural myocardial injury, PCAT in chronic coronary syndrome, post-hoc analyses in T2DM cohorts.

### 7.12 European Groups

- **Amsterdam UMC (Coerkamp, Henriques)**: CaRi-Heart clinical deployment studies, risk reclassification (62% of patients reclassified)
- **Mannheim (Ayx, Froelich, Nörenberg)**: PCD-CT radiomic texture analysis of PCAT on NAEOTOM Alpha
- **Leiden**: Innovative imaging techniques
- **Politecnico di Milano (Nannini, Redaelli)**: PCAT as predictor of functional severity (CT-FFR correlation)

### 7.13 MolloiLab / UCI (Our Group)

**Core Assumption**: *"If the measurement is fundamentally confounded, don't correct it — replace it."*

**Key investigators**: Sabee Molloi, Shu Nie

| Design Choice | Rationale | Blind Spot |
|---|---|---|
| XCAT phantom simulation | Complete ground truth control — known tissue composition | Simulation ≠ clinical reality; must be validated in vivo |
| Multi-energy material decomposition (water, lipid, collagen, iodine) | Physics-based separation eliminates protocol dependence | Requires DECT or PCD-CT hardware |
| Protocol-independence as primary metric | Directly addresses the field's central limitation | Clinical endpoints not yet studied |
| Comparison across kVp and patient sizes | Demonstrate protocol independence systematically | Does not yet demonstrate superiority in MACE prediction |

**DNA Summary**: MolloiLab's approach is a **paradigm shift** rather than an incremental improvement. Rather than correcting for confounders within the HU domain, material decomposition exits the HU domain entirely. The current study extends the previous paper (Nie & Molloi, *Int J Cardiovasc Imaging* 2025) to show that inflamed PCAT (more water, less lipid) is detectable through composition but not through protocol-dependent HU.

**Key papers**: Ding et al. 2021; Nie S, Molloi S. *Int J Cardiovasc Imaging* 2025;41:1091–1101

### 7.14 Comparative Summary

| Group | Signal Assumption | Validation Philosophy | Generalisability Strategy | Measurement Scope |
|---|---|---|---|---|
| **Oxford** | Bio >> tech variance | Outcome prediction (MACE/death) | Scale (40,091 patients) | FAI → FAI Score → AI-Risk |
| **Erlangen** | Tech ≈ bio variance | Physics-first skepticism | Distance analysis, PCD-CT | Raw PCAT attenuation |
| **Cedars-Sinai** | Signal in texture features | Rigorous ML with internal validation | Protocol matching | 93–1,103 radiomic features |
| **Zurich** | Must characterise instrument | Within-subject repeated measures | Phantom + in vivo | Kernel/reconstruction effects |
| **Case Western** | Foundation must be stable | Temporal stability analysis | Perfusion timing control | FAI + 93 radiomic features |
| **Groningen** | Quantify what you can't fix | Normative reference data | Per-vessel, per-kV calibration | Reference FAI values |
| **Tohoku** | FAI = general inflammation | PET validation (gold standard) | Non-atherosclerotic diseases | PET + CT correlation |
| **ShuKun/China** | Scale reveals patterns | Large multicentre cohorts | Multi-centre, multi-population | Lesion-specific radiomics |
| **MolloiLab** | HU domain is fundamentally limited | Ground-truth simulation | Protocol-independent by design | Material composition |

---

## Part VIII: Emerging Directions

### 8.1 PCAT Radiomics: Beyond Mean HU

#### Rationale

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

#### Key Radiomics Studies (2024–2026)

- **Shang et al. (2025, *Cardiovasc Diabetol*, n=777, multicentre)**: PCAT radiomics improved MACE prediction beyond traditional risk scores. Combined model C-index: 0.873 (training), 0.824 (validation). Significant reclassification improvement (NRI: 0.256–0.480).
- **Hou/Liu et al. (2024, *Insights Imaging*, n=180)**: PCAT radiomic signature predicted rapid plaque progression. First-order statistics and higher-order texture features were most predictive.
- **Huang et al. (2025, *Front Cardiovasc Med*)**: Compared lesion-specific vs proximal PCAT radiomics models for MACE — lesion-specific approach showed superior performance.
- **PCAT radiomics for INOCA in NAFLD (2025, *BMC Cardiovasc Disord*)**: Radiomics model (AUC 0.734) outperformed simple PCAT attenuation (AUC 0.674) for diagnosing ischaemia with non-obstructive coronary arteries in NAFLD patients.

#### Critical Limitation: Radiomic Feature Instability

> Wu et al. (2025, Case Western): **78% of radiomic features change >10%** between different contrast perfusion phases.

This means radiomic models trained on one acquisition timing may not generalise to different timing protocols — the same protocol-dependence problem as FAI, but amplified across 93 features.

### 8.2 Photon-Counting CT (PCD-CT) and Spectral Imaging

#### What PCD-CT Offers

PCD-CT (Siemens NAEOTOM Alpha, GE Revolution CT) provides simultaneous multi-energy data:
- VMI at any keV (40–190 keV)
- Material decomposition maps (water, iodine, lipid, calcium)
- Effective atomic number (Z-eff) maps
- Ultra-high resolution mode (0.2 mm pixels)

#### Key PCD-CT Studies

- **Mergen V et al. (*AJR* 2021)**: First systematic assessment of EAT/FAI on PCD-CT. Phantom and in vivo (n=30). VMI at 55–80 keV compared to reference 120 kVp EID scan. Fat attenuation varies significantly with VMI energy level — 70 keV VMI approximates 120 kVp but is not identical.
- **Engel et al. (*J Clin Med* 2026)**: First study applying the −70.1 HU FAI threshold on PCD-CT. FAI ≥ −70.1 HU identified more lipid-rich, non-calcified plaques (vulnerable morphology).
- **Tremamunno G et al. (*Acad Radiol* 2025)**: Intra-individual FAI differences between PCD-CT and conventional EID CT confirmed measurements are NOT directly comparable. However, iterative reconstruction minimises most differences, enabling inter-scanner comparability.
- **Kravchenko D et al. (*Int J Cardiol* 2025)**: Extended the Tremamunno FAI work to full radiomic analysis — compared PCAT radiomic feature stability between PCD-CT and EID-CT within the same patients.
- **Kahmann J et al. (*Front Cardiovasc Med* 2024)**: PCAT texture analysis and CAD characterization on PCD-CT. Explored radiomic features of LAD and RCA PCAT on the NAEOTOM Alpha platform.
- **Gao et al. (2025, *Eur J Radiol*)**: PCD-CT UHR mode significantly reduced stent blooming artifacts. Stent-specific FAI was lower in PCD-CT vs simulated conventional CT.
- **2025 (*Front Cardiovasc Med*)**: Development of NEW threshold for pericoronary fat attenuation based on **40 keV VMI** from dual-energy spectral CT — recognizing the old threshold doesn't transfer.

#### VMI Considerations

VMI at 70 keV closely matches conventional 120 kVp CT in noise characteristics, but fat HU values shift approximately **+5 to +15 HU** due to energy-dependent attenuation differences. The −70.1 HU threshold has not been validated on VMI data. The Zurich group (Alkadhi, Eberhard, Mergen) has systematically evaluated these shifts.

---

## Part IX: Commercial Landscape

### 9.1 Summary

| Company | FAI Analysis | Plaque Analysis | AI Risk Score | FDA Cleared | Reimbursement |
|---|---|---|---|---|---|
| **Caristo (CaRi-Heart)** | ✅ Core product | ✅ CaRi-Plaque | ✅ CaRi-Heart score | ✅ (2025) | ✅ Medicare 2026 |
| **Cleerly** | ❌ | ✅ Core product | ✅ | ✅ | ✅ Category I CPT |
| **HeartFlow** | ❌ | ✅ | ✅ (with FFR) | ✅ | ✅ Category I CPT |
| **ShuKun** | ✅ Radiomics | ✅ | Partial | Regional (China) | Regional |

### 9.2 Caristo Diagnostics (CaRi-Heart) — Oxford Spinoff

- **Product**: CaRi-Heart (FAI-based risk) + CaRi-Plaque™ (AI plaque analysis)
- **FDA 510(k) clearance**: CaRi-Plaque™ (March 2025)
- **CPT codes**: AMA assigned Category III codes 0992T and 0993T (2025)
- **Medicare reimbursement**: Finalized across hospital and office settings starting 2026
- **Clinical deployment**: First U.S. hospital (NCH) implementing CaRi-Heart
- **NHS study (2024)**: Demonstrated CaRi-Heart could reduce cardiac deaths by 12% in the UK NHS
- **Key advantage**: Only commercial tool with FAI + AI risk score, backed by CRISP-CT and ORFAN data

### 9.3 Cleerly

- **Focus**: AI-based comprehensive coronary plaque analysis (not specifically FAI-focused)
- **Funding**: $106M round (2024)
- **CPT**: Category I code for AI-QCT advanced plaque analyses
- **Coverage**: Aetna, UnitedHealthcare, Cigna, Humana — 86+ million lives
- **Distinction**: Emphasises plaque characterisation (stenosis, composition, remodelling) over pericoronary inflammation

### 9.4 HeartFlow

- **Product**: HeartFlow Plaque Analysis, FDA 510(k) cleared (2025)
- **CPT**: New Category I code for AI-enabled plaque analysis
- **Funding**: $890.5M total
- **Focus**: CT-FFR + plaque analysis, not specifically PCAT/FAI

### 9.5 ShuKun Technology (China)

- **Product**: Peri-coronary Adipose Tissue Analysis Tool + CoronaryDoc®-FFR
- **Funding**: $296.4M total
- **Focus**: PCAT radiomics (93 features), lesion-specific analysis
- **Market**: Strong presence in Asian markets, used in multiple Chinese multicentre studies

---

## Part X: Research Trajectory Timeline

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
| 2026 | **Current study**: XCAT + material decomposition for PCAT | Nie S, Molloi (UCI) | *In preparation* |

---

## Part XI: Study Design Patterns and Patient Selection

### 11.1 Study Design Patterns

| Study Type | Proportion | Examples |
|---|---|---|
| **Retrospective cohort** | ~60% | CRISP-CT, most single-centre PCAT studies |
| **Prospective cohort** | ~15% | ORFAN, some NAEOTOM Alpha studies |
| **Meta-analysis / systematic review** | ~10% | Sagris et al. 2022, Tan et al. 2025 |
| **Phantom / simulation** | ~10% | Etter et al. 2022, Nie & Molloi 2025 |
| **Case-control** | ~5% | ACS vs. stable angina comparisons |

### 11.2 PCAT Measurement Protocol

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

### 11.3 Patient Selection

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

### 11.4 Patient Data in Key Studies

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

### 11.5 Outcome-Based Cohort Design

For prognostic PCAT studies, the dominant design pattern is:

1. **Retrospective identification** of patients who had CCTA at a specific centre/time period
2. **Follow-up** via medical records / national registries for MACE events (MI, cardiac death, revascularisation, heart failure hospitalisation)
3. **Minimum follow-up**: typically ≥2 years (ORFAN: 2.7 years median; CRISP-CT: 5 years)
4. **Sample size**: typically n=200–500 for single-centre; n>1,000 for multicentre/registry
5. **Event rate**: ~5–15% at 5 years in unselected CCTA populations (higher in ACS cohorts)
6. **Statistical approach**: Cox proportional hazards with FAI as continuous or dichotomised (−70.1 HU) variable, adjusted for age, sex, cardiovascular risk factors, CACS, and stenosis severity

---

## Part XII: Material Decomposition — Our Approach

### 12.1 Core Argument

All of the confounders described in Parts IV–VI affect **HU values** (the physical measurement underlying FAI) but do NOT affect the **actual tissue composition** (water, lipid, protein, iodine content). Material decomposition — decomposing each voxel into its constituent materials — is inherently protocol-independent because it measures composition, not attenuation.

In the unified framework (Part IV), material decomposition is the only approach that **eliminates g entirely** rather than correcting for it:

```
HU-based approaches:    measure f + g → try to remove g → estimate f̂ ≠ f
Material decomposition:  measure f directly → f̂ ≈ f (protocol-independent)
```

### 12.2 Key Results from Nie & Molloi 2025

> Nie S, Molloi S. "Quantification of Water and Lipid Composition of Perivascular Adipose Tissue Using Coronary CT Angiography: A Simulation Study." *Int J Cardiovasc Imaging* 2025;41:1091–1101.

- Water fraction RMSE: 0.01–0.64% (sufficient to detect the ~5% clinical threshold)
- HU variance across 80–135 kV: **21.9%** (protocol-dependent)
- HU variance across patient sizes: **3.6%** (protocol-dependent)
- **Material decomposition (water fraction) was protocol-independent**: same composition yielded same water fraction regardless of kV or patient size

### 12.3 Current XCAT Study

The current study extends this work by using anatomically realistic XCAT phantoms, simulating pericoronary adipose inflammation as increased water content, and decomposing into 4 materials (water, lipid, collagen, iodine). The key demonstration: FAI (HU) differs across protocols for the same tissue, but material decomposition gives consistent composition regardless of protocol.

### 12.4 Positioning in the Field

For multi-site trials, longitudinal monitoring, and cross-scanner comparisons, a protocol-independent biomarker is essential. The field is increasingly recognising this need (Part V §5.10), and material decomposition provides a fundamentally different approach than the calibration/correction strategies (conversion factors, FAI Score) currently proposed.

Our approach directly addresses each of the four paradoxes identified in Part VI §6.6:
1. **Oxford-Erlangen**: Material decomposition measures f without g — resolving whether the signal is biological or artifactual
2. **Correction Paradox**: No regression-based correction needed — signal is measured directly
3. **Clinical Paradox**: Composition-based measurement may detect treatment effects that HU-based FAI misses
4. **Radiomics Paradox**: Composition maps provide protocol-independent features for texture analysis

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
