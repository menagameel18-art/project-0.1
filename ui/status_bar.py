"""
Workstation Status Bar with Loading Progress & Inspector Metadata.
"""
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QStatusBar, QLabel, QProgressBar, QWidget, QHBoxLayout, QFrame
)
from imaging.volume_processor import SeriesMetadata, VolumeGeometry


class WorkstationStatusBar(QStatusBar):
    """Clinical status bar with real-time cursor inspection and system indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setStyleSheet("""
            QStatusBar {
                background-color: #121214;
                color: #999999;
                border-top: 1px solid #252528;
                font-size: 11px;
            }
            QStatusBar::item {
                border: none;
            }
        """)

        self._setup_widgets()

    def _setup_widgets(self):
        # Left status / message
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #00d2ff; font-weight: bold; padding-left: 6px;")
        self.addWidget(self.lbl_status)

        # Loading progress bar (permanent widget)
        self.p_bar = QProgressBar()
        self.p_bar.setFixedSize(140, 14)
        self.p_bar.setRange(0, 100)
        self.p_bar.setVisible(False)
        self.p_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333338;
                border-radius: 2px;
                text-align: center;
                font-size: 9px;
                background-color: #1a1a1d;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #007acc;
            }
        """)
        self.addPermanentWidget(self.p_bar)

        # Active Patient & Series info
        self.lbl_patient = QLabel("Patient: None")
        self.lbl_patient.setStyleSheet("color: #e0e0e0; padding: 0 8px;")
        self.addPermanentWidget(self.lbl_patient)

        # Dimensions & Spacing
        self.lbl_geometry = QLabel("Vol: -")
        self.lbl_geometry.setStyleSheet("color: #aaaaaa; padding: 0 8px;")
        self.addPermanentWidget(self.lbl_geometry)

        # Cursor Inspector
        self.lbl_inspector = QLabel("Cursor: -")
        self.lbl_inspector.setStyleSheet("color: #ffcc00; font-weight: 500; padding: 0 8px;")
        self.addPermanentWidget(self.lbl_inspector)

        # Engine status
        self.lbl_engine = QLabel("VTK 3D Engine")
        self.lbl_engine.setStyleSheet("color: #4ade80; font-weight: bold; padding-right: 8px;")
        self.addPermanentWidget(self.lbl_engine)

    def show_progress(self, current: int, total: int, msg: str = ""):
        self.lbl_status.setText(msg or "Processing...")
        self.p_bar.setVisible(True)
        if total > 0:
            pct = int((current / total) * 100)
            self.p_bar.setValue(min(100, pct))

    def hide_progress(self, msg: str = "Ready"):
        self.p_bar.setVisible(False)
        self.lbl_status.setText(msg)

    def set_metadata(self, meta: SeriesMetadata, geom: VolumeGeometry):
        p_name = meta.patient_name.replace("^", " ")
        self.lbl_patient.setText(f"Patient: {p_name} ({meta.patient_id}) | {meta.modality} - {meta.series_description}")
        self.lbl_geometry.setText(f"Vol: {geom.num_x}x{geom.num_y}x{geom.num_z} ({geom.dx:.2f}x{geom.dy:.2f}x{geom.dz:.2f} mm)")

    def update_inspector(self, plane: str, col: float, row: float, val: Optional[float]):
        if val is not None:
            self.lbl_inspector.setText(f"[{plane}] ({int(col)}, {int(row)}) = {val:.1f} HU")
        else:
            self.lbl_inspector.setText("Cursor: Outside Volume")
