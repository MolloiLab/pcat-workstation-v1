# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PCAT Workstation.

Build with:
    pyinstaller pcat_workstation.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Paths
ROOT = Path(SPECPATH)
APP_DIR = ROOT / "pcat_workstation"
PIPELINE_DIR = ROOT / "pipeline"

# Collect all application modules
a = Analysis(
    [str(APP_DIR / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Include pipeline module as data (imported at runtime)
        (str(PIPELINE_DIR), "pipeline"),
    ],
    hiddenimports=[
        # Qt/VTK
        "vtkmodules",
        "vtkmodules.all",
        "vtkmodules.qt.QVTKRenderWindowInteractor",
        "vtkmodules.vtkRenderingOpenGL2",
        "vtkmodules.vtkRenderingFreeType",
        "vtkmodules.vtkInteractionStyle",
        # scipy submodules
        "scipy.ndimage",
        "scipy.interpolate",
        "scipy.signal",
        "scipy.spatial",
        # scikit-image
        "skimage",
        "skimage.measure",
        "skimage.segmentation",
        # matplotlib backends
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_pdf",
        # PySide6
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        # TotalSegmentator (optional, for seed generation)
        "totalsegmentator",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PCAT Workstation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,  # macOS double-click support
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PCAT Workstation",
)

# macOS .app bundle
app = BUNDLE(
    coll,
    name="PCAT Workstation.app",
    icon=None,  # TODO: add app icon
    bundle_identifier="edu.uci.molloi.pcat-workstation",
    info_plist={
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleName": "PCAT Workstation",
    },
)
