"""
Professional Medical Case Library Dialog.
Allows users to explore, filter, download, and load benchmark clinical imaging studies.
"""
from __future__ import annotations
from typing import List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QScrollArea, QWidget, QPushButton, QProgressBar, QFrame, QMessageBox
)

from dicom.sample_data_manager import SampleDataManager, SampleCase


class SampleCaseCard(QFrame):
    """UI card displaying details for a single sample case with actions."""
    open_requested = Signal(str)
    download_requested = Signal(str)

    def __init__(self, case: SampleCase, parent=None):
        super().__init__(parent)
        self.case = case
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            SampleCaseCard {
                background-color: #1a1a1d;
                border: 1px solid #2d2d33;
                border-radius: 6px;
                padding: 10px;
            }
            SampleCaseCard:hover {
                border-color: #007acc;
                background-color: #1e1e22;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(14)

        # Modality Badge
        badge_color = "#007acc" if self.case.modality == "CT" else "#8a2be2"
        badge = QLabel(self.case.modality)
        badge.setFixedSize(48, 48)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(f"""
            background-color: {badge_color};
            color: #ffffff;
            font-weight: bold;
            font-size: 14px;
            border-radius: 6px;
        """)
        layout.addWidget(badge)

        # Center Details
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        title_lbl = QLabel(self.case.title)
        title_lbl.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 13px;")
        info_layout.addWidget(title_lbl)

        sub_lbl = QLabel(f"Region: {self.case.anatomical_region}  •  Size: {self.case.size_str}  •  License: {self.case.license_info}")
        sub_lbl.setStyleSheet("color: #00d2ff; font-size: 11px;")
        info_layout.addWidget(sub_lbl)

        desc_lbl = QLabel(self.case.description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #b0b0b8; font-size: 11px;")
        info_layout.addWidget(desc_lbl)

        layout.addLayout(info_layout, stretch=1)

        # Right Action Column
        action_layout = QVBoxLayout()
        action_layout.setSpacing(6)
        action_layout.setAlignment(Qt.AlignCenter)

        self.status_lbl = QLabel("Ready" if self.case.is_downloaded else "Not Downloaded")
        status_color = "#4ade80" if self.case.is_downloaded else "#f59e0b"
        self.status_lbl.setStyleSheet(f"color: {status_color}; font-weight: bold; font-size: 11px;")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        action_layout.addWidget(self.status_lbl)

        if self.case.is_downloaded or self.case.is_procedural_demo:
            btn_open = QPushButton("Open Case")
            btn_open.setFixedSize(110, 30)
            btn_open.setStyleSheet("""
                QPushButton {
                    background-color: #007acc;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #0098ff;
                }
            """)
            btn_open.clicked.connect(lambda: self.open_requested.emit(self.case.id))
            action_layout.addWidget(btn_open)
        else:
            btn_dl = QPushButton("Download")
            btn_dl.setFixedSize(110, 30)
            btn_dl.setStyleSheet("""
                QPushButton {
                    background-color: #2e7d32;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #388e3c;
                }
            """)
            btn_dl.clicked.connect(lambda: self.download_requested.emit(self.case.id))
            action_layout.addWidget(btn_dl)

        layout.addLayout(action_layout)


class SampleLibraryDialog(QDialog):
    """Full-featured modal catalog for browsing and downloading medical cases."""
    case_opened = Signal(str)

    def __init__(self, sample_mgr: SampleDataManager, parent=None):
        super().__init__(parent)
        self.sample_mgr = sample_mgr
        self.setWindowTitle("Medical Case Library - Benchmark Datasets & Demonstrations")
        self.resize(850, 620)
        self.setStyleSheet("background-color: #121214; color: #ffffff;")

        self._setup_ui()
        self._connect_manager()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        # Header Title
        title = QLabel("Clinical DICOM & Volumetric Sample Case Library")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00d2ff;")
        main_layout.addWidget(title)

        subtitle = QLabel("Explore real human anatomical CT and MRI benchmark studies with complete multiplanar and 3D rendering.")
        subtitle.setStyleSheet("color: #888890; font-size: 11px;")
        main_layout.addWidget(subtitle)

        # Filter Bar
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)

        filter_layout.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter by title, anatomy, or description...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e24;
                border: 1px solid #33333d;
                border-radius: 4px;
                padding: 6px 10px;
                color: white;
            }
        """)
        self.search_box.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.search_box, stretch=1)

        filter_layout.addWidget(QLabel("Modality:"))
        self.combo_modality = QComboBox()
        self.combo_modality.addItems(["All", "CT", "MR"])
        self.combo_modality.setStyleSheet("""
            QComboBox {
                background-color: #1e1e24;
                border: 1px solid #33333d;
                border-radius: 4px;
                padding: 4px 8px;
                color: white;
            }
        """)
        self.combo_modality.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.combo_modality)

        filter_layout.addWidget(QLabel("Region:"))
        self.combo_region = QComboBox()
        self.combo_region.addItems(["All", "Head / Brain", "Chest / Thorax", "Abdomen", "Skeletal / Bone"])
        self.combo_region.setStyleSheet("""
            QComboBox {
                background-color: #1e1e24;
                border: 1px solid #33333d;
                border-radius: 4px;
                padding: 4px 8px;
                color: white;
            }
        """)
        self.combo_region.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.combo_region)

        main_layout.addLayout(filter_layout)

        # Scroll Area for Cards
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: 1px solid #25252b; background-color: #141416; }")

        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(8, 8, 8, 8)
        self.card_layout.setSpacing(10)
        self.scroll.setWidget(self.card_container)
        main_layout.addWidget(self.scroll, stretch=1)

        # Download Progress Section (initially hidden)
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        self.progress_frame.setStyleSheet("background-color: #1e1e24; border: 1px solid #2d2d38; border-radius: 4px; padding: 8px;")
        p_layout = QVBoxLayout(self.progress_frame)

        self.p_label = QLabel("Downloading...")
        self.p_label.setStyleSheet("color: #00d2ff; font-weight: bold;")
        p_layout.addWidget(self.p_label)

        p_row = QHBoxLayout()
        self.p_bar = QProgressBar()
        self.p_bar.setRange(0, 100)
        self.p_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #33333d;
                border-radius: 3px;
                text-align: center;
                background-color: #121214;
            }
            QProgressBar::chunk {
                background-color: #007acc;
            }
        """)
        p_row.addWidget(self.p_bar, stretch=1)

        self.btn_cancel_dl = QPushButton("Cancel")
        self.btn_cancel_dl.setFixedSize(70, 24)
        self.btn_cancel_dl.setStyleSheet("background-color: #c62828; color: white; border-radius: 3px;")
        self.btn_cancel_dl.clicked.connect(self.sample_mgr.cancel_download)
        p_row.addWidget(self.btn_cancel_dl)

        p_layout.addLayout(p_row)
        main_layout.addWidget(self.progress_frame)

        # Bottom Buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setFixedSize(100, 32)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #2b2b30;
                color: white;
                border-radius: 4px;
                border: 1px solid #3d3d45;
            }
            QPushButton:hover {
                background-color: #383842;
            }
        """)
        btn_close.clicked.connect(self.accept)
        bottom_layout.addWidget(btn_close)

        main_layout.addLayout(bottom_layout)

        self._populate_cards()

    def _connect_manager(self):
        self.sample_mgr.case_download_started.connect(self._on_download_started)
        self.sample_mgr.case_download_progress.connect(self._on_download_progress)
        self.sample_mgr.case_download_finished.connect(self._on_download_finished)
        self.sample_mgr.case_download_failed.connect(self._on_download_failed)

    def _populate_cards(self):
        # Clear existing
        while self.card_layout.count():
            item = self.card_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        cases = self.sample_mgr.get_cases()
        search = self.search_box.text().strip().lower()
        sel_mod = self.combo_modality.currentText()
        sel_reg = self.combo_region.currentText()

        for c in cases:
            # Filter checks
            if sel_mod != "All" and c.modality != sel_mod:
                continue
            if sel_reg != "All" and sel_reg not in c.anatomical_region:
                continue
            if search and (search not in c.title.lower() and search not in c.description.lower() and search not in c.anatomical_region.lower()):
                continue

            card = SampleCaseCard(c)
            card.open_requested.connect(self._on_open_case)
            card.download_requested.connect(self._on_start_download)
            self.card_layout.addWidget(card)

        self.card_layout.addStretch()

    def _apply_filters(self):
        self._populate_cards()

    def _on_open_case(self, case_id: str):
        self.case_opened.emit(case_id)
        self.accept()

    def _on_start_download(self, case_id: str):
        self.sample_mgr.download_case(case_id)

    def _on_download_started(self, title: str):
        self.progress_frame.setVisible(True)
        self.p_label.setText(f"Downloading: {title}...")
        self.p_bar.setValue(0)

    def _on_download_progress(self, dl_bytes: int, total_bytes: int, msg: str):
        self.p_label.setText(msg)
        if total_bytes > 0:
            pct = int((dl_bytes / total_bytes) * 100)
            self.p_bar.setValue(min(100, pct))

    def _on_download_finished(self, case_id: str, local_dir: str):
        self.progress_frame.setVisible(False)
        self._populate_cards()
        QMessageBox.information(self, "Download Complete", "The medical case dataset has been downloaded and validated.")

    def _on_download_failed(self, case_id: str, error_msg: str):
        self.progress_frame.setVisible(False)
        QMessageBox.warning(self, "Download Failed", f"Could not complete dataset download: {error_msg}")
