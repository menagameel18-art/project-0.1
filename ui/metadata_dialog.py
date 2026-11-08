"""
Patient and Study DICOM Metadata Inspector Dialog.
"""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton
)
from imaging.volume_processor import VolumeProcessor


class MetadataDialog(QDialog):
    """Displays comprehensive DICOM metadata table."""

    def __init__(self, processor: VolumeProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.setWindowTitle("DICOM Patient & Study Metadata")
        self.resize(650, 480)
        self.setStyleSheet("background-color: #161619; color: #ffffff;")

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("DICOM Acquisition & Geometric Parameters")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #00d2ff;")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #121214;
                color: #e0e0e0;
                gridline-color: #2b2b32;
                border: 1px solid #2b2b32;
            }
            QHeaderView::section {
                background-color: #1f1f24;
                color: #00d2ff;
                font-weight: bold;
                border: 1px solid #2b2b32;
                padding: 4px;
            }
        """)

        # Populate attributes
        meta = self.processor.metadata
        geom = self.processor.geometry

        attributes = [
            ("Patient Name", meta.patient_name.replace("^", " ")),
            ("Patient ID", meta.patient_id),
            ("Study Date", meta.study_date),
            ("Modality", meta.modality),
            ("Study Description", meta.study_description),
            ("Series Description", meta.series_description),
            ("Series Instance UID", meta.series_uid),
            ("Study Instance UID", meta.study_uid),
            ("Number of Slices (Z)", str(geom.num_z if geom else meta.num_slices)),
            ("Slice Matrix (X x Y)", f"{geom.num_x} x {geom.num_y}" if geom else "-"),
            ("Voxel Spacing (dx, dy, dz)", f"{geom.dx:.3f} mm, {geom.dy:.3f} mm, {geom.dz:.3f} mm" if geom else "-"),
            ("Physical Origin (X0, Y0, Z0)", f"({geom.origin[0]:.2f}, {geom.origin[1]:.2f}, {geom.origin[2]:.2f}) mm" if geom else "-"),
            ("Slice Normal Vector", f"({geom.slice_normal[0]:.3f}, {geom.slice_normal[1]:.3f}, {geom.slice_normal[2]:.3f})" if geom else "-"),
            ("Rescale Slope", str(meta.rescale_slope)),
            ("Rescale Intercept", str(meta.rescale_intercept)),
            ("CT Modality HU Calibrated", "Yes (Hounsfield Units)" if meta.is_ct else "No (Intensity Scaled)")
        ]

        self.table.setRowCount(len(attributes))
        for row, (k, v) in enumerate(attributes):
            item_k = QTableWidgetItem(k)
            item_k.setFlags(item_k.flags() ^ Qt.ItemIsEditable)
            item_v = QTableWidgetItem(v)
            item_v.setFlags(item_v.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_k)
            self.table.setItem(row, 1, item_v)

        layout.addWidget(self.table, stretch=1)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setFixedSize(90, 30)
        btn_close.setStyleSheet("background-color: #007acc; color: white; border-radius: 4px; font-weight: bold;")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)
