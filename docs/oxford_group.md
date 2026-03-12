# Oxford Group — Pericoronary Adipose Tissue Research Program

> **PI: Prof. Charalambos Antoniades, MD PhD FRCP FESC**
> BHF Chair of Cardiovascular Medicine, University of Oxford
> Founder & Director, Caristo Diagnostics Ltd.

---

## Table of Contents

- [How They Built This: Research Architecture](#how-they-built-this-research-architecture)
- [Phase 1: The Biology (2013–2016)](#phase-1-the-biology-20132016)
- [Phase 2: Conceptual Framework Building (2017 Reviews)](#phase-2-conceptual-framework-building-2017-reviews)
- [Phase 3: FAI — From Biology to Imaging (2017–2018)](#phase-3-fai--from-biology-to-imaging-20172018)
- [Phase 4: Extending the Science (2019)](#phase-4-extending-the-science-2019)
- [Phase 5: Clinical Translation (2020–2023)](#phase-5-clinical-translation-20202023)
- [Phase 6: Population-Scale Validation (2024–2026)](#phase-6-population-scale-validation-20242026)
- [The Patent](#the-patent)
- [Caristo Diagnostics](#caristo-diagnostics)
- [Open Questions and Limitations](#open-questions-and-limitations)
- [Knowledge Gaps to Fill](#knowledge-gaps-to-fill)
- [Paper Catalogue](#paper-catalogue)

---

## How They Built This: Research Architecture

The Oxford program spans 13 years and follows a coherent logic: establish the biology in human tissue, translate it to an imaging readout, validate the imaging in large cohorts, standardize it for clinical use, and build the regulatory/economic evidence for adoption.

What makes it work is the tight coupling between bench science and clinical imaging. Each imaging claim traces back to a specific molecular mechanism demonstrated in human tissue. This is rare — most imaging biomarkers are discovered empirically (statistical association with outcomes) and only later investigated mechanistically, if at all. The Oxford group worked the other direction: they understood why PVAT changes before they ever measured it on a CT scan.

**The CABG tissue pipeline** is central to everything. Oxford's cardiac surgery service provides:
- Internal mammary arteries with intact PVAT (ex vivo vascular experiments)
- Saphenous veins for vasomotor studies
- Matched adipose tissue from 4 depots: perivascular (peri-IMA, peri-SV), epicardial, thoracic, subcutaneous
- Preoperative CT imaging for imaging-histology correlation

Cumulative tissue biobank across papers: n=677 (2013), n=386 (2015), n=453+37+273 (2017), n=167 (2019 FRP), n=1,004 (2019 WNT5A), n=1,200+ (2023 miR-92a-3p). This is thousands of matched tissue-imaging pairs, built up over a decade.

**How each paper connects to the next:**

```
2013 Adiponectin/eNOS → Outside-to-inside signaling established
  ↓ But what about NADPH oxidase specifically? And disease states?
2015 Adiponectin/NADPH oxidase in T2D → Mendelian randomization, coculture proof
  ↓ But this is arteries only. What about the myocardium?
2016 EAT-myocardium signaling → Inside-to-outside loop completed
  ↓ PVAT composition reflects vascular disease. Can imaging detect this?
2017 FAI discovery → CT captures inflammation-induced PVAT changes
  ↓ Cross-sectional only. Does it predict future events?
2018 CRISP-CT → FAI predicts cardiac mortality in 3,912 patients
  ↓ FAI is dynamic/reversible. What about chronic structural damage?
2019 FRP → Radiomic texture captures fibrosis + vascularity
  ↓ What molecular pathways beyond adiponectin drive PVAT dysfunction?
2019 WNT5A → Second druggable axis identified
  ↓ How to standardize FAI across scanners for clinical use?
2021 FAI-Score → Scanner-adjusted, age/sex-normalized metric
  ↓ Does it hold at scale? Is it cost-effective?
2024 ORFAN → 40,091-patient validation
2025 Cost-effectiveness → ICER £1,371-3,244/QALY
```

No paper repeats the previous one. Each answers a specific limitation and opens a new question.

---

## Phase 1: The Biology (2013–2016)

### Margaritis et al. 2013 — Circulation | n=677+46 CABG | ~321 citations

This paper established the core idea: PVAT and the vascular wall communicate bidirectionally through paracrine signals.

**Methods worth understanding in detail:**
- **Brachial FMD measured the day before surgery** — gives an in vivo endothelial function measure to correlate with ex vivo tissue findings. This bridge between clinical and bench data is a recurring design element.
- **Lucigenin chemiluminescence (5 μmol/L)** for vascular superoxide. The lower concentration reduces redox cycling artifacts inherent to higher doses.
- **HPLC for biopterins (BH4/BH2)** — quantifies eNOS cofactor availability, which determines whether eNOS produces NO (coupled) or superoxide (uncoupled). This assay is technically demanding and specific to their group's expertise.
- **Two ADIPOQ SNPs (rs17366568, rs266717)** used as Mendelian randomization instruments — genetic variants that alter circulating adiponectin levels, providing quasi-causal evidence without the confounding of observational data.
- **4-HNE incubation of PVAT (30 μmol/L, 16h)** ± PPARγ inhibitor T0070907 — tests the reverse signal (vessel → fat).

**The 8-step logical chain:**
1. Circulating adiponectin correlates with endothelium-dependent (but not -independent) vascular function — establishing specificity
2. Circulating adiponectin inversely correlates with vascular O2⁻, BUT local PVAT adiponectin positively correlates — a paradox that reveals two different regulatory systems
3. Circulating adiponectin tracks with remote fat depots (mesothoracic, subcutaneous) but NOT perivascular — confirming different regulation
4. PPARγ is the master regulator of ADIPOQ across all depots (r values 0.344-0.976)
5. ADIPOQ SNPs predict circulating adiponectin and endothelial function, but do NOT predict PVAT adiponectin — genetic evidence applies only to the systemic pathway
6. Exogenous adiponectin restores eNOS coupling in ex vivo vessels
7. Dual mechanism: PI3K/Akt-mediated eNOS phosphorylation (blocked by wortmannin) + BH4 biosynthesis (blocked by DAHP)
8. Vascular oxidative stress (4-HNE) upregulates PVAT adiponectin via PPARγ — the reverse signal

**What was new:** The bidirectional cross-talk concept. Previous work had shown PVAT releases adipokines that affect vessels (outside-to-inside). Nobody had shown the vessel signals back to the fat (inside-to-outside). The dissociation between circulating and local adiponectin — showing they have opposite associations with vascular O2⁻ — was a genuine insight that reframed the field.

**Honest limitations:**
- 4-HNE at 30 μmol/L is pharmacological; physiological relevance at the tissue interface is assumed
- No demonstration that adiponectin (30kDa+ multimer) actually diffuses from PVAT across the adventitia into the vessel wall — paracrine delivery is assumed but never measured
- 83% male; no sex-stratified analysis
- The Mendelian randomization validates only the circulating pathway, not the local paracrine pathway they emphasize

---

### Antonopoulos et al. 2015 — Diabetes | n=386+67 CABG | ~241 citations

Extended the 2013 work to NADPH oxidase specifically and to the disease context of type 2 diabetes.

**Key methodological advances over 2013:**
- **Vas2870** (specific NADPH oxidase inhibitor) allows isolation of NADPH oxidase-derived superoxide from total superoxide
- **Thoracic AT (non-vessel-associated)** added as a control depot — cleaner comparison than subcutaneous
- **Coculture system:** PVAT incubated with/without its matched IMA, ± NADPH stimulation, ± PEG-SOD. This directly tested whether vascular NADPH oxidase activity signals to neighboring PVAT.

**The finding that matters most:** In multivariable analysis including ADIPOQ genotype, T2D loses independent prediction of NADPH oxidase activity (β=0.068, P=0.192) while ADIPOQ genotype remains significant (β=-0.081, P=0.001). This is genuine Mendelian randomization evidence that T2D's effect on vascular oxidative stress is mediated through adiponectin.

**Important negative finding:** Plasma MDA and 4-HNE (systemic oxidative stress markers) do NOT correlate with arterial NADPH oxidase-derived O2⁻ (r=0.076, P=0.448). This is one of the most consequential observations in the entire program — it means systemic blood biomarkers cannot substitute for local tissue assessment. This finding recurs throughout their work and is the fundamental justification for imaging-based approaches.

**Two temporal mechanisms of adiponectin on NADPH oxidase:** Rapid (6h) RAC1 deactivation preventing NADPH oxidase assembly at the membrane, and subacute (18h) p22phox protein downregulation. This level of mechanistic detail is what gives their imaging claims biological weight.

---

### Antonopoulos et al. 2016 — Circulation Research | ~158 citations

Extended bidirectional signaling from the PVAT-artery axis to the EAT-myocardium axis. Myocardial oxidative stress generates 4-HNE → diffuses into epicardial fat → activates PPARγ → upregulates adiponectin as a protective feedback loop.

This bridged from peripheral vessels to the cardiac compartment, justifying later expansion into EAT quantification and EAT-derived miRNA biology.

---

### What Phase 1 teaches us

The biology rests on a specific experimental toolkit: organ bath physiology, lucigenin chemiluminescence, HPLC biopterins, Western blotting for signaling intermediates, DHE fluorescence microscopy, ex vivo coculture, and Mendelian randomization via adipokine gene variants. Understanding these methods — their strengths and their artifacts — is essential for evaluating and extending the work.

The CABG patient population constrains generalizability. These are patients with advanced multi-vessel CAD, mean age ~65, 83% male, nearly all on statins and ACE inhibitors. Whether bidirectional PVAT-vascular signaling operates similarly in early disease, younger patients, or women is an open question.

---

## Phase 2: Conceptual Framework Building (2017 Reviews)

Five reviews in a single year, across 4 journals. Their function was to synthesize the group's biological findings into a conceptual framework and identify the field's unresolved problems — problems their upcoming work would address.

| Paper | Journal | Core Contribution |
|---|---|---|
| "Epicardial AT in cardiac biology" | J Physiol (5.5) | Comprehensive EAT biology reference: embryology, thermogenesis, mechanical buffering, paracrine signaling |
| "Is fat always bad?" | Cardiovasc Res (10.2) | Argued fat has protective roles that become pathological through "reprogramming" — the conceptual basis for FAI |
| "Dysfunctional adipose tissue" | BJP (7.3) | Introduced the concept that adipose dysfunction is a reprogrammable therapeutic target |
| "PVAT as regulator of vascular disease" | BJP (7.3) | Mapped the full PVAT secretome and signaling pathways; identified therapeutic targets |
| "Unravelling the adiponectin paradox" | BJP (7.3) | Resolved why high adiponectin associates with poor outcomes in HF: it's a compensatory response to vascular injury, not a cause |

The adiponectin paradox resolution is worth understanding: epidemiological studies found elevated adiponectin in patients with worse cardiovascular outcomes, contradicting its known protective biology. The Oxford group's answer — that vascular oxidative stress upregulates PVAT adiponectin as a defense mechanism (inside-to-outside signaling) — means elevated adiponectin in sick patients is a marker of the rescue response, not the disease itself. This is a clean example of mechanistic biology resolving an epidemiological contradiction.

These reviews also systematically evaluated competing approaches. EAT volume studies (Framingham, MESA, Rotterdam) had produced inconsistent results, partly because they treated EAT as a homogeneous depot. PET-CT could detect tissue inflammation but with poor spatial resolution and limited availability. Circulating biomarkers (hsCRP, IL-6) lacked vessel-specificity. These were real limitations of existing methods, identified clearly.

---

## Phase 3: FAI — From Biology to Imaging (2017–2018)

### Antonopoulos et al. 2017 — Science Translational Medicine | ~872 citations

The conceptual leap: if vascular inflammation changes PVAT adipocyte biology (inhibiting differentiation, reducing lipid accumulation), and CT attenuation depends on the lipid-to-water ratio in tissue, then CT can non-invasively detect vascular inflammation by measuring the attenuation of perivascular fat.

**Multi-arm study design — each arm answers a different question:**
- **Arm 1 (n=453 CABG):** Histological validation — paired tissue biopsies and CT, showing that FAI correlates with adipocyte size, differentiation markers (PPARγ, CEBPA, FABP4), and macrophage infiltration
- **Arm 2 (n=37 CABG):** Coculture proof — aortic tissue pre-treated with angiotensin II (7 days) then cocultured with PVAT preadipocytes inhibits their lipid accumulation. TNFα + IL-6 + IFNγ directly suppress adipocyte differentiation genes.
- **Arm 3 (n=273 CTA subjects):** Clinical validation — FAI higher around culprit lesions in ACS, decreases on follow-up after stenting. FAI predicts CAD independently of calcium score.
- **PET sub-study (n=40):** ¹⁸F-FDG uptake in subcutaneous AT correlates with FAI (ρ=0.69, AUC=0.971). Note: this validated in subcutaneous but not epicardial fat — an important distinction.

**Measurement protocol defined here:**
- Attenuation window: -190 to -30 HU
- PVAT = adipose tissue within radial distance equal to vessel diameter from outer wall
- RCA: 10-50mm from ostium (excludes proximal 10mm to avoid aortic wall artifact)
- LAD and LCx: proximal 40mm
- Analyzed in 3D concentric 1mm layers radiating outward

**VPCI (Volumetric Perivascular Characterization Index):** A gradient-based metric — the difference between PVAT attenuation and non-PVAT attenuation, normalized by PVAT attenuation. This self-normalizes against systemic metabolic factors because each patient's remote fat serves as their own reference. VPCI was actually superior to raw FAI for detecting soft (noncalcified) plaques.

**What was demonstrated vs. what was inferred:**
- Demonstrated: inflammation inhibits preadipocyte differentiation, FAI correlates with tissue biology, FAI differs between diseased and normal vessels
- Inferred but not yet proven: that the CT signal specifically reflects inflammation (vs. other processes that alter adipocyte biology), and that FAI changes predict future events

---

### Oikonomou et al. 2018 — The Lancet (CRISP-CT) | n=3,912 | ~838 citations

Post-hoc analysis of two prospective cohorts: Erlangen (n=1,872, 2005-2009) and Cleveland Clinic (n=2,040, 2008-2016). Five different CT scanners. All images analyzed blindly at one core lab (OXACCT, Oxford).

**Key results:**
| Metric | Derivation | Validation |
|---|---|---|
| Per-SD FAI (RCA), HR cardiac mortality | 2.15 (1.33-3.48) | 2.06 (1.50-2.83) |
| FAI ≥ -70.1 HU, HR cardiac mortality | 9.04 (3.35-24.40) | 5.62 (2.90-10.88) |
| C-statistic improvement | 0.913 → 0.962 | 0.763 → 0.838 |
| NRI for cardiac mortality | 0.94 | 0.72 |
| Technical parameters R² for FAI | ~0.05 | — |
| ICC (intra/inter-observer) | 0.987 / 0.980 | — |

**The J-shaped relationship** is noteworthy: fractional polynomial modeling showed non-linear FAI-mortality association. Extremely negative FAI (very fatty PVAT) may also be pathological — potentially reflecting lipomatous metaplasia or loss of normal adipocyte architecture. This motivated dichotomization at -70.1 HU (Youden's J statistic) rather than treating FAI as purely linear.

**Limitations to understand:**
1. The -70.1 HU cutoff was derived from this data and needed prospective validation (came in 2024 ORFAN)
2. Only 26 cardiac deaths in the German derivation cohort (1.4%) — large hazard ratios from few events
3. Among patients who started statins/aspirin after CCTA, FAI lost prognostic significance (HR 2.85, P=0.25). This could mean the risk is "treatable" — or that FAI captures the same risk traditional therapy already addresses
4. No comparison with hsCRP or any blood biomarker
5. FAI's dynamism (it changes after ACS and with treatment) is both a strength and a limitation — it may be too labile for stable risk assessment

---

## Phase 4: Extending the Science (2019)

### FRP — Oikonomou et al. 2019, European Heart Journal | ~400 citations

FAI captures acute inflammation (the lipid/water shift in PVAT). But PVAT also undergoes chronic structural changes — fibrosis and microvascular remodeling — that are irreversible. Mean attenuation cannot detect these. Texture analysis can.

**Radiotranscriptomic approach (Study 1, n=167 surgery patients):**
Paired CT radiomic features with tissue gene expression for three biological processes:
- TNFα → inflammation
- COL1A1 → fibrosis
- CD31 → vascularity

Result: Mean attenuation (≈FAI) was the best CT predictor of TNFα. But higher-order texture features matched or exceeded it for COL1A1 and CD31. Adding radiomics significantly improved detection of fibrosis (P=0.005) and vascularity (P=0.015) but NOT inflammation (P=0.35). This directly proves that texture features capture biology invisible to FAI.

**ML pipeline:**
- 843 radiomic features per vessel × 2 vessels = 1,686 features/patient (PyRadiomics via 3D Slicer)
- Stability filtering (ICC ≥ 0.9): 1,391 features retained
- Correlation filtering (|Spearman ρ| ≥ 0.9): 335 independent features
- Recursive feature elimination + random forest: 64 optimal features
- Case-control training: 101 MACE cases matched 1:1 with 101 controls
- External AUC: 0.774 (0.622-0.926)

**Clinical validation (SCOT-HEART, n=1,575):**
- FRP ≥ 0.63: adjusted HR 10.84 (5.06-23.22) for MACE
- FRP+ with high-risk plaque features: HR 43.33 (9.14-205.48)
- FRP+ without high-risk plaque: HR 32.44 (7.00-150.38) — identifies risk with no visible plaque pathology
- FRP did NOT predict non-cardiac mortality (HR 0.58, P=0.28) — cardiac-specific

**Temporal dissociation between FAI and FRP:** In AMI patients with serial CT, FAI was elevated acutely around culprit lesions and decreased at 6-month follow-up (dynamic). FRP was elevated at both timepoints (stable/irreversible). This implies FAI tracks active inflammation while FRP tracks cumulative structural damage — complementary, not redundant.

**Limitations:** 101 events for training a 335-feature model is small. SCOT-HEART had only 34 MACE events (1 cardiac death). The radiotranscriptomic tissue was thoracic fat from the surgical incision, not pericoronary fat. The 64-feature random forest is not interpretable at the individual feature level.

---

### Psoriasis as a Model — Elnabawi/Antoniades 2019, JAMA Cardiology | ~197 citations

n=134 patients with moderate-to-severe psoriasis: 82 on biologics (anti-TNF, anti-IL-12/23, anti-IL-17), 52 untreated controls. CCTA at baseline and 1 year.

Why this population is well-suited: chronic systemic inflammation with elevated CV risk, low traditional CV risk factors (isolates the inflammatory signal), built-in control group, measurable treatment response via PASI score, and specific cytokine-targeted therapies enabling mechanistic dissection.

FAI decreased from -71.22 to -76.09 HU in the treatment group (P<0.001); no change in controls (-71.98 to -72.66, P=0.39). Effect present with both anti-TNF and anti-IL pathways.

This established FAI as a **treatment-response biomarker** — it can detect pharmacologically-induced changes in coronary inflammation non-invasively. This has applications in clinical trials for anti-inflammatory drugs.

---

### WNT5A — Akoumianakis et al. 2019, Science Translational Medicine

Their largest tissue study (n=1,004). Profiled all 19 Wnt ligands across 3 adipose depots and identified WNT5A as the most highly expressed in PVAT. Complete signaling pathway mapped: WNT5A → FZD2 receptor → USP17 (novel finding) → RAC1 activation → NOX1/2 assembly → superoxide generation.

The USP17 deubiquitinase link was entirely new to vascular biology. Dose-dependent pathway specificity: physiological WNT5A (100 ng/mL) activates non-canonical PCP pathway; only supraphysiological doses activate canonical Wnt/β-catenin. This resolves prior contradictions about WNT5A being pro- vs. anti-atherogenic.

Clinically, plasma WNT5A independently associated with CAD and predicted coronary calcification progression over 3-5 years. This diversified the group's molecular portfolio beyond adiponectin and created a second potentially druggable pathway. The 2023 miR-92a-3p paper later showed that miR-92a-3p suppresses WNT5A — connecting the two pathways.

---

## Phase 5: Clinical Translation (2020–2023)

### FAI + High-Risk Plaque Interaction (2020, JACC)

Re-analysis of CRISP-CT data in a 2×2 stratification: FAI (high/low at -70.1 HU) × high-risk plaque features (present/absent).

The key finding: **HRP without elevated FAI carries no excess cardiac mortality risk** (HR 1.00, P=0.98). But high FAI without HRP carries substantial risk (HR 5.62, P<0.001). The highest risk is when both are present (HR 7.29, P<0.001).

This implies that high-risk plaque morphology without active inflammation may represent stable remodeling, while inflammation — even without visible plaque pathology — identifies patients at risk. This challenges the plaque-centric paradigm in CCTA interpretation.

Limitation: post-hoc analysis of the same dataset, with few events per subgroup (74 cardiac deaths split across 4 cells).

### FAI-Score Standardization (2021, Cardiovascular Research)

Raw FAI depends on scanner, tube voltage, contrast protocol, patient age/sex, and specific artery. FAI-Score transforms raw FAI through age/sex-specific nomograms adjusted for technical parameters. Output as population-referenced percentiles (0-100 scale).

The CaRi-Heart device pipeline:
1. Deep learning segments EAT and perivascular space
2. Human analyst reviews and edits segmentations (not fully automated)
3. Raw FAI computed for proximal RCA, LAD, LCx
4. FAI-Score calculated (adjusted)
5. CaRi-Heart Risk: 8-year cardiac mortality probability integrating FAI-Score + clinical risk factors + plaque burden

AUC 0.809 for 8-year cardiac mortality. Optimism-corrected AUC = 0.809 (no overfitting detected). Trained on US cohort, externally validated in European cohort.

Important: all 3,912 scans analyzed by one core lab (OXACCT). Real-world reproducibility when community sites use the device is untested.

### Deep Learning EAT (2023, JACC Cardiovascular Imaging)

3D Residual-U-Net trained on 2,800 ORFAN CCTAs. Processing: 12.4 seconds vs. 18 minutes manual. Positioned as complementary to FAI: EAT volume predicts all-cause mortality including non-cardiac (metabolic marker), while FAI predicts cardiac-specific events. Different risk dimensions from the same scan.

ORFAN infrastructure: 75,000 UK patients across 17 NHS Trusts, expanding to 250,000 internationally.

### ESC Consensus Statement (2023, European Heart Journal)

The ESC Working Group endorsed:
- FAI-Score as the regulatory-cleared metric for coronary inflammation
- The Oxford PVAT definition (radial distance = vessel diameter from outer wall)
- CaRi-Heart v2.5 by name
- 75th, 90th, 95th percentile cutoffs for risk stratification

Gaps identified by the consensus: (1) no RCTs of FAI-guided therapy, (2) per-lesion FAI not validated, (3) photon-counting CT needs recalibration, (4) non-coronary PVAT not validated, (5) no PVAT-specific drug delivery.

### miR-92a-3p (2023, JACC)

n>1,200 across 6 study arms (GWAS, animal models, cell culture, clinical outcomes). Identified miR-92a-3p as an EAT-derived microRNA that reduces myocardial NADPH oxidase by suppressing WNT5A/RAC1. This extended the PVAT-vascular paracrine model to include miRNA mediators — a new mechanistic layer connecting back to the WNT5A pathway.

---

## Phase 6: Population-Scale Validation (2024–2026)

### ORFAN Study — Chan et al. 2024, The Lancet | n=40,091

**Cohort A:** 40,091 consecutive patients from 8 NHS hospitals (2010-2021), median follow-up 2.7 years. **Cohort B:** 3,393 nested patients with 7.7-year follow-up for AI-Risk validation. Events: 4,307 MACE, 1,754 cardiac deaths.

**Key results:**

| Measure | HR (95% CI) |
|---|---|
| LAD FAI-Score Q4 vs Q1, cardiac mortality | 20.20 (11.49-35.53) |
| 3 inflamed arteries vs 0, cardiac mortality | 29.8 (13.9-63.9) |
| AI-Risk very high vs low/medium, cardiac mortality | 6.75 (5.17-8.82) |

Discrimination: QRISK3 alone AUC=0.784 → +CAD-RADS 2.0: 0.789 (P=0.38, no improvement) → +AI-Risk: 0.854 (P=7.7×10⁻⁷). Adding stenosis grading to traditional risk scores did not improve prediction. Adding inflammation assessment did.

The central epidemiological finding: 81.1% of patients had no obstructive CAD, yet this group accounted for 66.3% of cardiac deaths. Current practice sends these patients home reassured. FAI-Score identifies the high-risk subset within them.

Real-world NHS survey (n=744): AI-Risk changed management in 45% — 24% new statin, 13% statin dose increase, 8% additional therapies.

Calibration: well-calibrated in non-obstructive CAD. Overestimates risk in obstructive CAD (because CCTA triggers interventions). Median follow-up of 2.7 years in Cohort A is short for an 8-year prediction model — the model extrapolates.

### Cost-Effectiveness (2025)

Markov model, 3,393 patients. AI-guided strategy: ICER £1,371-3,244/QALY. Predicted 11% MI reduction, 12% cardiac death reduction. 100% of probabilistic sensitivity analyses below the NICE £20,000 threshold. Designed for NICE technology appraisal submission.

### FAI-Score Robustness (2025)

n=7,822 CCTAs from one ORFAN site. FAI-Score stable within 0.5 units (on 0-100 scale) across tube voltage, tube current, slice thickness, and scan phase. Validates the standardization pipeline. Contrast protocol variations (injection rate, iodine concentration, scan timing) not explicitly tested.

---

## The Patent

### US 10,695,023 B2 | Filed 2015, Granted 2020

**4 independent claims**, all requiring: CT data → concentric layer analysis from outer vessel wall → radiodensity quantification per layer → comparison to baseline → administering therapy.

| Claim | Specific method |
|---|---|
| 1 (broadest) | Concentric layer volumetric characterization + therapy |
| 14 | VPCI-i: fold change plot → AUC calculation |
| 21 | VPCI: PVAT minus non-PVAT radiodensity |
| 37 | Treatment guidance: CT → concentric layers → therapy |

**Key dependent claims:** 4cm proximal RCA segment (claim 4), specific arteries including non-coronary (claims 5-6), 1mm layers (claim 7), various end-distance and baseline definitions (claims 8-13).

**What the patent does NOT cover:**
- Simple mean attenuation without concentric layer analysis
- Non-CT modalities (MRI, ultrasound, PET)
- Pure research/diagnostic use without the "administering therapy" step
- Machine learning radiomic approaches (FRP)
- FAI-Score nomogram/percentile calculation
- Photon-counting CT material decomposition approaches

Related applications: PCT/GB2017/053262, GB2018/1818049.7, GR20180100490, GR20180100510

---

## Caristo Diagnostics

| Product | Function | Status |
|---|---|---|
| **CaRi-Heart** (FAI-Score + AI-Risk) | Coronary inflammation from routine CCTA | CE Mark (MDR), UKCA, Australia. Investigational in US. |
| **CaRi-Plaque** | Automated plaque + stenosis quantification | CE Mark, UKCA, **FDA 510(k) cleared** (K242240, 2025) |
| **AI-Risk** | Integrated 8-year cardiac risk score | Part of CaRi-Heart platform |

Founders: Antoniades, Shirodaria, Channon, Neubauer. Data infrastructure: ORFAN (75,000+ UK, expanding to 250,000 internationally).

Note on independence: the entire evidence chain — from biomarker discovery to device development to clinical validation to consensus guidelines to health economics — has been produced by a group with direct financial interest in the product. Each paper discloses this. Independent validation by groups with no Caristo ties would substantially strengthen the evidence.

---

## Open Questions and Limitations

### Scientific

1. **Generalizability of the biology.** All mechanistic work comes from CABG patients: advanced CAD, ~83% male, mean age ~65, nearly all on statins. Does bidirectional PVAT-vascular signaling operate the same way in early subclinical disease? In women? In younger patients? In non-European populations?

2. **Adiponectin diffusion problem.** A 30kDa+ multimeric protein is assumed to traverse the adventitia from PVAT to the vascular media/endothelium. This has never been directly demonstrated. The paracrine mechanism may work through smaller mediators or through exosomes/miRNAs rather than intact adiponectin protein.

3. **FAI specificity for inflammation.** The biological model is: inflammation → cytokine-mediated inhibition of preadipocyte differentiation → less lipid accumulation → higher CT attenuation. But other processes also alter adipocyte biology (fibrosis, edema, hemorrhage, metabolic stress). FAI may capture a composite signal, not purely inflammation.

4. **The LCx problem.** LCx is excluded from most analyses due to variable anatomy and small caliber. But LCx disease causes real clinical events. Any PVAT imaging approach limited to RCA and LAD misses a major coronary territory.

5. **PVAT browning/beige fat.** PVAT can undergo brown-to-white conversion in disease states. UCP1, β3-adrenergic receptors, and thermogenic capacity represent a parallel biology that the Oxford group has not deeply explored. Brown vs. white fat has distinct imaging characteristics (CT and MRI).

### Translational

6. **No RCT evidence.** The ESC consensus identifies this as Gap #1. All evidence is observational. The 45% management change in ORFAN is uncontrolled — whether FAI-guided decisions actually improve outcomes is unknown.

7. **Proprietary algorithm.** The FAI-to-FAI-Score transformation and the AI-Risk classifier are unpublished. Independent groups cannot reproduce or verify the computations. This is standard for commercial medical devices but limits scientific scrutiny.

8. **Standardization treadmill.** Each new scanner generation (especially photon-counting CT), contrast protocol variation, and reconstruction algorithm potentially requires recalibration. The 2025 robustness data is from one site; multi-site multi-vendor robustness at scale is still developing.

9. **Short follow-up in the largest study.** ORFAN Cohort A median follow-up is 2.7 years for an 8-year prediction model. Substantial extrapolation is required.

---

## Knowledge Gaps to Fill

To work at the level of this program, the following domains need to be understood in depth — not as a checklist, but as interconnected knowledge:

**Vascular biology:** eNOS coupling/uncoupling (BH4 cofactor chemistry), NADPH oxidase isoforms and assembly (NOX1/2/4/5, RAC1/p47phox translocation), adipokine signaling (adiponectin receptors, WNT5A/FZD2 non-canonical pathway, PPARγ regulation), and redox signaling (4-HNE as a diffusible oxidation product). These aren't background — they're the specific molecular mechanisms behind every imaging claim.

**CT physics of fat imaging:** Why the -190 to -30 HU window works (lipid phase ≈ -190 HU, aqueous phase ≈ -30 HU, fat tissue sits on this spectrum based on adipocyte lipid content). How tube voltage shifts the attenuation curve. How reconstruction kernels affect texture features. How contrast timing affects pericoronary measurements. How partial volume effects at the vessel-fat interface create artifacts.

**Radiomics methodology:** Feature types (first-order statistics vs. GLCM/GLRLM/GLSZM texture matrices vs. wavelet transforms), stability analysis (ICC filtering), dimensionality reduction strategies, appropriate sample sizes for feature-to-event ratios, and the critical distinction between radiomic features that correlate with biology vs. features that are measurement artifacts.

**Survival statistics:** Cox proportional hazards, time-dependent C-statistics, NRI/IDI for censored data, decision curve analysis (net clinical benefit), fractional polynomials for non-linear relationships, and competing risks modeling. Every clinical paper uses these. Understanding them well enough to identify where statistical choices affect conclusions (e.g., the -70.1 HU cutoff was data-derived, not pre-specified).

**Mendelian randomization:** The Oxford group uses ADIPOQ SNPs as genetic instruments for causal inference. Understanding instrumental variable assumptions (relevance, independence, exclusion restriction), two-sample MR design, and pleiotropy tests is necessary to evaluate their causal claims.

**Regulatory and health economics:** CE marking (MDR) and FDA 510(k)/De Novo pathways for software as a medical device (SaMD). Markov decision-analytic models, ICER/QALY calculations, NICE technology appraisal requirements. These are the mechanics of clinical translation.

**Where opportunities lie:**
- **MRI-based PVAT characterization.** The patent is CT-specific. MRI offers fat/water fraction, T1/T2 mapping, diffusion-weighted imaging — potentially richer tissue characterization without ionizing radiation. No group has demonstrated PVAT inflammation detection by MRI.
- **Non-coronary PVAT.** Carotid (stroke), aortic (aneurysm), femoral (PAD) — all clinically important, all acknowledged as unvalidated in the ESC consensus.
- **Open-source reproducibility.** A transparent, publicly validated PVAT analysis pipeline would be scientifically complementary to the proprietary approach and attract collaborators.
- **Photon-counting CT / dual-energy CT.** Material decomposition enables direct fat/water/calcium separation — a fundamentally different measurement physics from conventional single-energy attenuation.
- **Diverse populations.** Women, non-white cohorts, younger patients with early disease.
- **Prospective interventional evidence.** Even a small RCT of PVAT-guided therapy would address the field's most significant evidence gap.

---

## Paper Catalogue

### Phase 1: Biology (2013–2016)

| # | Year | Title | Journal (IF) | Cit. | Key Contribution |
|---|------|-------|-------------|------|-----------------|
| 1 | 2013 | Adiponectin/eNOS in Human Vessels | *Circulation* (35.5) | ~321 | Bidirectional PVAT-vascular signaling; dual adiponectin mechanism. n=677+46. |
| 2 | 2014 | Systemic Inflammation and BNP on Adiponectin | *ATVB* (8.4) | — | BNP upregulates adiponectin; links HF neurohormones to fat. |
| 3 | 2015 | Adiponectin Links T2D to NADPH Oxidase | *Diabetes* (7.7) | ~241 | NADPH oxidase-specific; Mendelian randomization. n=386+67. |
| 4 | 2016 | EAT-Myocardial Redox via PPARγ/Adiponectin | *Circ Res* (20.1) | ~158 | Inside-to-outside signaling in cardiac compartment. |

### Phase 2: Reviews (2017)

| # | Year | Title | Journal (IF) | Cit. |
|---|------|-------|-------------|------|
| 5 | 2017 | EAT in cardiac biology | *J Physiol* (5.5) | ~162 |
| 6 | 2017 | Is fat always bad? | *Cardiovasc Res* (10.2) | ~148 |
| 7 | 2017 | Dysfunctional adipose tissue | *BJP* (7.3) | — |
| 8 | 2017 | PVAT as vascular disease regulator | *BJP* (7.3) | ~73 |
| 9 | 2017 | Adiponectin paradox | *BJP* (7.3) | ~134 |

### Phase 3: FAI & CRISP-CT (2017–2018)

| # | Year | Title | Journal (IF) | Cit. | Key Contribution |
|---|------|-------|-------------|------|-----------------|
| **10** | **2017** | **Detecting coronary inflammation by imaging PVAT** | ***Sci Transl Med*** **(17.1)** | **~872** | **FAI invention.** n=453+37+273. |
| **11** | **2018** | **CRISP-CT** | ***Lancet*** **(168.9)** | **~838** | **Prognostic validation.** n=3,912. HR 9.04 cardiac mortality. |
| 12 | 2018 | Adipose tissue in CV health and disease | *Nat Rev Cardiol* (41.7) | ~396 | Definitive review positioning FAI. |
| 13 | 2018 | PVAT and coronary atherosclerosis | *Heart* (5.0) | ~113 | Bridge paper: biology → imaging. PVAT definition. |

### Phase 4: Expansion (2019)

| # | Year | Title | Journal (IF) | Cit. | Key Contribution |
|---|------|-------|-------------|------|-----------------|
| **14** | **2019** | **Radiotranscriptomic FRP** | ***EHJ*** **(39.3)** | **~400** | **Radiomic texture captures fibrosis + vascularity beyond FAI.** |
| 15 | 2019 | Biologic therapy in psoriasis | *JAMA Cardiol* (14.8) | ~197 | FAI as treatment-response biomarker. n=134. |
| 16 | 2019 | Imaging residual inflammatory CV risk | *EHJ* (39.3) | ~145 | Clinical positioning review with Deanfield. |
| 17 | 2019 | Atherosclerosis affecting fat | *JCCT* (3.3) | ~102 | FAI methodology review. |
| 18 | 2019 | CT Assessment of Coronary Inflammation | *ATVB* (8.4) | ~35 | Technical review. |
| 19 | 2019 | Making Sense From Perivascular Attenuation Maps | *JACC CVI* (12.8) | ~33 | Editorial/interpretation guide. |
| 20 | 2019 | WNT5A/USP17/RAC1 pathway | *Sci Transl Med* (17.1) | — | Second druggable axis. n=1,004. |

### Phase 5: Translation (2020–2023)

| # | Year | Title | Journal (IF) | Cit. | Key Contribution |
|---|------|-------|-------------|------|-----------------|
| 21 | 2020 | FAI + HRP Stratification | *JACC* (21.7) | ~87 | HRP without inflammation = no excess risk. |
| 22 | 2020 | AI Radiomic Guide | *Cardiovasc Res* (10.2) | ~77 | 16-point radiomic quality framework. |
| 23 | 2021 | FAI in Coronary CTA | *Radiol Cardiothorac Imaging* (4.5) | ~62 | Practical FAI guide. |
| 24 | 2021 | Standardized FAI measurement | *Cardiovasc Res* (10.2) | ~33 | FAI-Score and CaRi-Heart specification. AUC 0.809. |
| 25 | 2021 | PVAT imaging by CT: virtual guide | *BJP* (7.3) | ~38 | Measurement protocols and pitfalls. |
| 26 | 2022 | FAI Meta-Analysis | *EHJ CVI* (6.2) | — | 20 studies, 7,797 patients. MACE HR 3.29. |
| 27 | 2022 | Pericardial Adiposity CMR | *EHJ CVI* (6.2) | ~22 | UK Biobank n=42,598. EAT volume → adverse remodeling, AF. |
| 28 | 2023 | Deep Learning EAT | *JACC CVI* (12.8) | ~77 | 3D Res-U-Net. 12.4s processing. Predicts mortality + AF. |
| 29 | 2023 | EAT-derived miR-92a-3p | *JACC* (21.7) | ~19 | miR-92a-3p suppresses WNT5A/RAC1 in myocardium. |
| 30 | 2023 | ESC Consensus Statement | *EHJ* (39.3) | — | ESC endorses FAI-Score, CaRi-Heart, PVAT definition. |

### Phase 6: Validation (2024–2026)

| # | Year | Title | Journal (IF) | Cit. | Key Contribution |
|---|------|-------|-------------|------|-----------------|
| **31** | **2024** | **ORFAN Study** | ***Lancet*** **(168.9)** | — | **n=40,091.** HR 20.20 cardiac mortality Q4 vs Q1. |
| 32 | 2024 | AI in atherosclerosis CT | *Atherosclerosis* (5.3) | — | Landscape review. |
| 33 | 2025 | FAI-Score robustness | — | — | Cross-scanner stability. n=7,822. |
| 34 | 2025 | FAI Score standardization editorial | *JACC CVI* (12.8) | — | Case for standardized measurement. |
| 35 | 2025 | Cost-effectiveness of AI | *EHJ QoCC* (4.6) | — | ICER £1,371-3,244/QALY. |
| 36 | 2026 | PVAT Imaging and Quantification | *ATVB* (8.4) | — | Latest methodological review. |

---

**Summary:** 36 papers | 19 original research | 12 reviews | 5 other (consensus, meta-analysis, editorial, health economics) | 1 US patent | 3 commercial products | ~50,000 patients validated | 8 publications in IF>15 journals

---

*Last updated: March 12, 2026*
