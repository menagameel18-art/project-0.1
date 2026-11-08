"""
First-Run Experience & Welcome Modal Dialog.
Provides fast onboarding with 1-click access to local DICOM folders,
curated sample libraries, and instant 3D demo cases.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)


class WelcomeDialog(QDialog):
    """First-run welcome screen offering fast entry points."""
    open_folder_requested = Signal()
    explore_samples_requested = Signal()
    load_demo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Medical DICOM 3D Workstation")
        self.resize(580, 420)
        self.setStyleSheet("background-color: #121216; color: #ffffff;")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(18)

        # Header Title
        title_lbl = QLabel("Medical Imaging & 3D Reconstruction Workstation")
        title_lbl.setStyleSheet("color: #00d2ff; font-size: 18px; font-weight: bold;")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        sub_lbl = QLabel("Professional multiplanar DICOM viewer (Axial, Sagittal, Coronal) and VTK 3D volume renderer.")
        sub_lbl.setStyleSheet("color: #9e9ea8; font-size: 12px;")
        sub_lbl.setAlignment(Qt.AlignCenter)
        sub_lbl.setWordWrap(True)
        layout.addWidget(sub_lbl)

        layout.addSpacing(10)

        # Action 1: Load Demo Case (Recommended / 1-Click)
        card_demo = self._create_action_card(
            title="Load Instant Demo Case (CT Head 3D)",
            desc="Explore all four synchronized viewports and 3D volume reconstruction immediately with zero download required.",
            btn_text="Launch Demo",
            btn_color="#007acc",
            callback=self._on_demo_clicked
        )
        layout.addWidget(card_demo)

        # Action 2: Open Local DICOM Folder
        card_folder = self._create_action_card(
            title="Open My DICOM Folder",
            desc="Select a directory containing CT, MRI, or medical DICOM series on your computer.",
            btn_text="Open Folder",
            btn_color="#107c41",
            callback=self._on_folder_clicked
        )
        layout.addWidget(card_folder)

        # Action 3: Explore Sample Cases
        card_samples = self._create_action_card(
            title="Explore Built-in Sample Cases",
            desc="Browse benchmark public medical datasets: CT Chest, CT Head, Abdomen, Bone, and Brain MRI.",
            btn_text="Browse Cases",
            btn_color="#5c2d91",
            callback=self._on_samples_clicked
        )
        layout.addWidget(card_samples)

        layout.addStretch()

        # Notice
        notice_lbl = QLabel("Antigravity Medical Imaging Software • Research & Visualization")
        notice_lbl.setStyleSheet("color: #555560; font-size: 10px;")
        notice_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(notice_lbl)

    def _create_action_card(self, title: str, desc: str, btn_text: str, btn_color: str, callback) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1a1a20;
                border: 1px solid #282832;
                border-radius: 6px;
                padding: 6px;
            }
            QFrame:hover {
                border-color: #007acc;
            }
        """)
        c_layout = QHBoxLayout(card)
        c_layout.setContentsMargins(12, 10, 12, 10)

        info_l = QVBoxLayout()
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 13px;")
        d_lbl = QLabel(desc)
        d_lbl.setStyleSheet("color: #888898; font-size: 11px;")
        d_lbl.setWordWrap(True)
        info_l.addWidget(t_lbl)
        info_l.addWidget(d_lbl)
        c_layout.addLayout(info_l, stretch=1)

        btn = QPushButton(btn_text)
        btn.setFixedSize(110, 32)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {btn_color};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: none;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """)
        btn.clicked.connect(callback)
        c_layout.addWidget(btn)

        return card

    def _on_demo_clicked(self):
        self.load_demo_requested.emit()
        self.accept()

    def _on_folder_clicked(self):
        self.open_folder_requested.emit()
        self.accept()

    def _on_samples_clicked(self):
        self.explore_samples_requested.emit()
        self.accept()
