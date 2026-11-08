"""
Main Application Top Toolbar.
Hosts primary clinical imaging tools, window/level presets, crosshair link toggle,
and study navigation actions.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QToolBar, QToolButton, QComboBox, QLabel, QWidget, QHBoxLayout,
    QButtonGroup
)


class MainToolBar(QToolBar):
    """Top toolbar for medical workstation."""
    open_folder_requested = Signal()
    sample_cases_requested = Signal()
    demo_case_requested = Signal()
    segmentation_panel_requested = Signal()
    preset_selected = Signal(str)
    tool_changed = Signal(str)           # 'window_level', 'zoom', 'pan', 'measure', 'crosshair', 'segment_brush', 'segment_eraser', 'segment_contour'
    crosshair_toggled = Signal(bool)
    reset_layout_requested = Signal()
    reset_views_requested = Signal()
    info_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setMovable(False)
        self.setFixedHeight(42)
        self.setStyleSheet("""
            QToolBar {
                background-color: #1a1a1d;
                border-bottom: 1px solid #28282e;
                spacing: 6px;
                padding: 3px 8px;
            }
        """)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        self._setup_actions()

    def _setup_actions(self):
        # 1. Open Folder Button
        btn_open = self._create_btn("Open Folder", "Open local DICOM folder (Ctrl+O)", self.open_folder_requested.emit)
        self.addWidget(btn_open)

        # 2. Sample Cases Button
        btn_samples = self._create_btn("Sample Cases", "Browse built-in Medical Case Library", self.sample_cases_requested.emit)
        btn_samples.setStyleSheet("""
            QToolButton {
                background-color: #1e3a5f;
                color: #00d2ff;
                border: 1px solid #007acc;
                border-radius: 4px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QToolButton:hover {
                background-color: #284e80;
            }
        """)
        self.addWidget(btn_samples)

        # 3. Quick Demo Button
        btn_demo = self._create_btn("Load Demo", "Load instant CT Head 3D demonstration", self.demo_case_requested.emit)
        self.addWidget(btn_demo)

        # 4. Segmentation Editor Button
        btn_seg = self._create_btn("Segmentation", "Open Segmentation Editor panel", self.segmentation_panel_requested.emit)
        btn_seg.setStyleSheet("""
            QToolButton {
                background-color: #2a1f42;
                color: #c084fc;
                border: 1px solid #7e22ce;
                border-radius: 4px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QToolButton:hover {
                background-color: #3b275e;
            }
        """)
        self.addWidget(btn_seg)

        self.addSeparator()

        # 5. Window / Level Preset Dropdown
        lbl_wl = QLabel(" W/L Preset: ")
        lbl_wl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.addWidget(lbl_wl)

        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "Soft Tissue / Abdomen",
            "Bone",
            "Lung",
            "Brain",
            "Mediastinum",
            "Liver",
            "Subdural / Blood"
        ])
        self.combo_presets.setStyleSheet("""
            QComboBox {
                background-color: #25252b;
                color: #ffffff;
                border: 1px solid #3d3d48;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
        """)
        self.combo_presets.currentTextChanged.connect(self.preset_selected.emit)
        self.addWidget(self.combo_presets)

        self.addSeparator()

        # 6. Interactive Mouse Tools (Exclusive Group)
        self.btn_wl_tool = self._create_toggle_tool("W/L", "Adjust Contrast & Brightness (Left-Drag)", "window_level")
        self.btn_pan_tool = self._create_toggle_tool("Pan", "Pan Image (Left-Drag)", "pan")
        self.btn_zoom_tool = self._create_toggle_tool("Zoom", "Zoom In / Out (Left-Drag)", "zoom")
        self.btn_measure_tool = self._create_toggle_tool("Measure", "Distance Caliper Tool (Click 2 points)", "measure")
        self.btn_crosshair_tool = self._create_toggle_tool("Crosshair", "Linked Crosshair Navigation (Click or Drag)", "crosshair")
        self.btn_brush_tool = self._create_toggle_tool("Brush", "Paint segmentation mask (B)", "segment_brush")
        self.btn_eraser_tool = self._create_toggle_tool("Eraser", "Erase segmentation mask (E)", "segment_eraser")
        self.btn_contour_tool = self._create_toggle_tool("Contour", "Draw closed contour region", "segment_contour")

        self.btn_wl_tool.setChecked(True)

        self.addWidget(self.btn_wl_tool)
        self.addWidget(self.btn_pan_tool)
        self.addWidget(self.btn_zoom_tool)
        self.addWidget(self.btn_measure_tool)
        self.addWidget(self.btn_crosshair_tool)
        self.addWidget(self.btn_brush_tool)
        self.addWidget(self.btn_eraser_tool)
        self.addWidget(self.btn_contour_tool)

        self.addSeparator()

        # 6. Crosshair Link Toggle
        self.btn_link = QToolButton()
        self.btn_link.setText("Linked Nav: ON")
        self.btn_link.setCheckable(True)
        self.btn_link.setChecked(True)
        self.btn_link.setStyleSheet("""
            QToolButton {
                background-color: #1b382b;
                color: #4ade80;
                border: 1px solid #2e7d32;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QToolButton:!checked {
                background-color: #25252b;
                color: #888888;
                border: 1px solid #3d3d45;
            }
        """)
        self.btn_link.toggled.connect(self._on_link_toggled)
        self.addWidget(self.btn_link)

        self.addSeparator()

        # 7. Layout and Reset Controls
        btn_reset_views = self._create_btn("Reset Views", "Reset 2D zooms and 3D camera", self.reset_views_requested.emit)
        btn_reset_layout = self._create_btn("Reset 2x2", "Reset 2x2 splitters to equal size", self.reset_layout_requested.emit)
        btn_info = self._create_btn("Study Info", "View Patient and Study DICOM Metadata", self.info_requested.emit)

        self.addWidget(btn_reset_views)
        self.addWidget(btn_reset_layout)
        self.addWidget(btn_info)

    def _create_btn(self, text: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QToolButton {
                background-color: #25252b;
                color: #e0e0e0;
                border: 1px solid #3d3d48;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QToolButton:hover {
                background-color: #32323a;
                border-color: #007acc;
            }
        """)
        btn.clicked.connect(callback)
        return btn

    def _create_toggle_tool(self, text: str, tooltip: str, tool_id: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setStyleSheet("""
            QToolButton {
                background-color: #25252b;
                color: #cccccc;
                border: 1px solid #383842;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QToolButton:checked {
                background-color: #007acc;
                color: #ffffff;
                font-weight: bold;
                border-color: #00aaff;
            }
            QToolButton:hover:!checked {
                background-color: #303038;
            }
        """)
        btn.clicked.connect(lambda: self.tool_changed.emit(tool_id))
        self._tool_group.addButton(btn)
        return btn

    def _on_link_toggled(self, checked: bool):
        self.btn_link.setText("Linked Nav: ON" if checked else "Linked Nav: OFF")
        self.crosshair_toggled.emit(checked)
