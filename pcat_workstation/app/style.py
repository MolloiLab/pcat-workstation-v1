"""Horos-inspired dark theme stylesheet for PCAT Workstation."""

COLORS = {
    "bg": "#1c1c1e",
    "panel": "#2c2c2e",
    "surface": "#3a3a3c",
    "text": "#e5e5e7",
    "text_secondary": "#98989d",
    "text_tertiary": "#636366",
    "border": "#38383a",
    "hover": "#3a3a3c",
    "selection": "#464649",
    "accent": "#0a84ff",
    "accent_hover": "#0070e0",
    "warning": "#ff9f0a",
    "success": "#30d158",
    "error": "#ff453a",
}


def get_stylesheet() -> str:
    """Return a complete QSS dark-theme stylesheet."""

    bg = COLORS["bg"]
    panel = COLORS["panel"]
    surface = COLORS["surface"]
    text = COLORS["text"]
    text_secondary = COLORS["text_secondary"]
    text_tertiary = COLORS["text_tertiary"]
    border = COLORS["border"]
    hover = COLORS["hover"]
    selection = COLORS["selection"]
    accent = COLORS["accent"]
    accent_hover = COLORS["accent_hover"]
    warning = COLORS["warning"]
    success = COLORS["success"]
    error = COLORS["error"]

    return f"""
/* ── Base ─────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {bg};
    color: {text};
    font-size: 15pt;
}}

/* ── QPushButton ──────────────────────────────────── */
QPushButton {{
    background-color: {accent};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    min-height: 36px;
    min-width: 36px;
    padding: 10px 20px;
    font-size: 15pt;
}}
QPushButton:hover {{
    background-color: {accent_hover};
}}
QPushButton:pressed {{
    background-color: {accent_hover};
}}
QPushButton:disabled {{
    background-color: {surface};
    color: {text_tertiary};
}}

/* Secondary buttons — apply via setProperty("class", "secondary") */
QPushButton[class="secondary"] {{
    background-color: transparent;
    color: {text};
    border: 1px solid {border};
}}
QPushButton[class="secondary"]:hover {{
    background-color: {hover};
}}
QPushButton[class="secondary"]:pressed {{
    background-color: {selection};
}}

/* ── QToolBar ─────────────────────────────────────── */
QToolBar {{
    background-color: {panel};
    border-bottom: 1px solid {border};
    spacing: 8px;
    padding: 4px;
}}
QToolBar::separator {{
    width: 1px;
    background-color: {border};
    margin: 4px 6px;
}}
QToolBar QToolButton {{
    min-width: 36px;
    min-height: 36px;
    border-radius: 6px;
    padding: 4px;
    color: {text};
}}
QToolBar QToolButton:hover {{
    background-color: {hover};
}}

/* ── QDockWidget ──────────────────────────────────── */
QDockWidget {{
    color: {text};
    border: 1px solid {border};
}}
QDockWidget::title {{
    background-color: {panel};
    padding: 8px 12px;
    border-bottom: 1px solid {border};
    font-size: 18pt;
    font-weight: bold;
}}

/* ── QTreeWidget / QListWidget ────────────────────── */
QTreeWidget, QListWidget {{
    background-color: {panel};
    color: {text};
    border: 1px solid {border};
    border-radius: 8px;
    alternate-background-color: {bg};
    outline: none;
    padding: 4px;
}}
QTreeWidget::item, QListWidget::item {{
    min-height: 36px;
    padding: 4px 8px;
    border-radius: 4px;
}}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background-color: {selection};
    color: {text};
}}
QTreeWidget::item:hover, QListWidget::item:hover {{
    background-color: {hover};
}}
QHeaderView::section {{
    background-color: {panel};
    color: {text_secondary};
    border: none;
    border-bottom: 1px solid {border};
    padding: 6px 8px;
    font-size: 13pt;
}}

/* ── QProgressBar ─────────────────────────────────── */
QProgressBar {{
    background-color: {surface};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {text_secondary};
    min-height: 18px;
    font-size: 13pt;
}}
QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 4px;
}}

/* ── QLabel ───────────────────────────────────────── */
QLabel {{
    color: {text};
    background-color: transparent;
}}

/* ── QScrollBar (vertical) ────────────────────────── */
QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {border};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {text_tertiary};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* ── QScrollBar (horizontal) ──────────────────────── */
QScrollBar:horizontal {{
    background-color: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background-color: {border};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {text_tertiary};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* ── QSplitter ────────────────────────────────────── */
QSplitter::handle {{
    background-color: {border};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── QGroupBox ────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px;
    padding-top: 20px;
    font-size: 15pt;
    background-color: {surface};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {text};
    font-size: 18pt;
    font-weight: bold;
}}

/* ── QTabWidget ───────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 8px;
    background-color: {bg};
}}
QTabBar::tab {{
    background-color: {panel};
    color: {text_secondary};
    padding: 10px 20px;
    border: none;
    border-bottom: 2px solid transparent;
    min-height: 36px;
    font-size: 15pt;
}}
QTabBar::tab:selected {{
    color: {text};
    border-bottom: 2px solid {accent};
    background-color: {bg};
}}
QTabBar::tab:hover:!selected {{
    background-color: {hover};
    color: {text};
}}

/* ── QStatusBar ───────────────────────────────────── */
QStatusBar {{
    background-color: {panel};
    color: {text_secondary};
    border-top: 1px solid {border};
    font-size: 13pt;
    padding: 4px 12px;
}}

/* ── QMenuBar / QMenu ─────────────────────────────── */
QMenuBar {{
    background-color: {panel};
    color: {text};
    border-bottom: 1px solid {border};
    padding: 2px;
}}
QMenuBar::item {{
    padding: 6px 12px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background-color: {hover};
}}
QMenu {{
    background-color: {bg};
    color: {text};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    min-height: 28px;
}}
QMenu::item:selected {{
    background-color: {hover};
}}
QMenu::separator {{
    height: 1px;
    background-color: {border};
    margin: 4px 8px;
}}

/* ── QToolTip ─────────────────────────────────────── */
QToolTip {{
    background-color: {text};
    color: {bg};
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 13pt;
}}

/* ── QComboBox ────────────────────────────────────── */
QComboBox {{
    background-color: {bg};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 36px;
    font-size: 15pt;
}}
QComboBox:hover {{
    border-color: {text_tertiary};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {text_secondary};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {bg};
    color: {text};
    border: 1px solid {border};
    border-radius: 8px;
    selection-background-color: {selection};
    selection-color: {text};
    outline: none;
    padding: 4px;
}}
"""
