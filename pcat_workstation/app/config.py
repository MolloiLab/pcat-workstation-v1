"""Configuration constants for PCAT Workstation."""

import os

# FAI thresholds (Antonopoulos et al., Sci Transl Med 2017)
FAI_HU_MIN = -190.0
FAI_HU_MAX = -30.0

# Risk threshold (Oikonomou et al., Lancet 2018, CRISP-CT)
FAI_RISK_THRESHOLD = -70.1  # HU, above this = HIGH risk (HR=9.04)

# Vessel segment definitions
VESSEL_CONFIGS = {
    "LAD": {"start_mm": 0.0, "length_mm": 40.0, "color": "#ff453a", "key": "1"},
    "LCx": {"start_mm": 0.0, "length_mm": 40.0, "color": "#0a84ff", "key": "2"},
    "RCA": {"start_mm": 10.0, "length_mm": 40.0, "color": "#30d158", "key": "3"},
}

# PCAT VOI geometry
DEFAULT_PCAT_SCALE = 3.0  # x mean vessel radius

# CT display defaults (vascular window for coronary artery work)
DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_LEVEL = 200

# Pipeline stages (ordered)
PIPELINE_STAGES = [
    "import",
    "seeds",
    "vesselness",
    "centerlines",
    "contours",
    "pcat_voi",
    "statistics",
]

STAGE_LABELS = {
    "import": "Loading volume",
    "seeds": "Detecting seeds (TotalSegmentator)",
    "vesselness": "Computing vesselness filter",
    "centerlines": "Extracting centerlines",
    "contours": "Extracting vessel contours",
    "pcat_voi": "Building PCAT VOI",
    "statistics": "Computing FAI statistics",
}

# Paths
DATA_DIR = os.path.expanduser("~/.pcat_workstation")
RECENT_PROJECTS_FILE = os.path.join(DATA_DIR, "recent_projects.json")
