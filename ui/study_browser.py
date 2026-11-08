"""
Collapsible Study & Series Browser Sidebar.
Displays patient/study hierarchy, series details, and quick-access sample cases.
"""
from __future__ import annotations
from typing import Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QHeaderView, QTabWidget, QListWidget, QListWidgetItem
)

from dicom.dicom_loader import DicomSeriesGroup
from dicom.sample_data_manager import SampleDataManager, SampleCase


class StudyBrowser(QWidget):
    """Sidebar tree browser for imported DICOM studies and built-in sample cases."""
    series_selected = Signal(str)      # series_uid
    sample_case_selected = Signal(str)  # case_id
    open_folder_requested = Signal()
    sample_library_requested = Signal()

    def __init__(self, sample_mgr: SampleDataManager, parent=None):
        super().__init__(parent)
        self.sample_mgr = sample_mgr
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Tab widget: "User Studies" and "Sample Library"
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2d2d2d;
                background-color: #18181a;
            }
            QTabBar::tab {
                background: #222224;
                color: #aaaaaa;
                padding: 6px 12px;
                border: 1px solid #2d2d2d;
                border-bottom: none;
                font-size: 11px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #18181a;
                color: #00d2ff;
                border-top: 2px solid #00d2ff;
            }
        """)

        # Tab 1: Imported DICOM Studies
        studies_page = QWidget()
        s_layout = QVBoxLayout(studies_page)
        s_layout.setContentsMargins(4, 4, 4, 4)
        s_layout.setSpacing(6)

        btn_open = QPushButton("Open DICOM Folder...")
        btn_open.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: #ffffff;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
        """)
        btn_open.clicked.connect(self.open_folder_requested.emit)
        s_layout.addWidget(btn_open)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Study / Series Description", "Mod", "Slices"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #121214;
                color: #e0e0e0;
                border: 1px solid #282828;
                border-radius: 3px;
                font-size: 11px;
            }
            QTreeWidget::item:hover {
                background-color: #26262a;
            }
            QTreeWidget::item:selected {
                background-color: #005999;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1f1f22;
                color: #888888;
                border: none;
                padding: 4px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self.tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        s_layout.addWidget(self.tree, stretch=1)

        self.btn_load_series = QPushButton("Load Selected Series")
        self.btn_load_series.setEnabled(False)
        self.btn_load_series.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2e;
                color: #e0e0e0;
                font-weight: bold;
                padding: 5px;
                border: 1px solid #3d3d40;
                border-radius: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #007acc;
                color: white;
            }
        """)
        self.btn_load_series.clicked.connect(self._on_load_selected_clicked)
        s_layout.addWidget(self.btn_load_series)

        self.tabs.addTab(studies_page, "DICOM Studies")

        # Tab 2: Sample Library Quick Access
        sample_page = QWidget()
        sam_layout = QVBoxLayout(sample_page)
        sam_layout.setContentsMargins(4, 4, 4, 4)
        sam_layout.setSpacing(6)

        btn_browse_lib = QPushButton("Explore Full Library...")
        btn_browse_lib.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #00d2ff;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
                border: 1px solid #007acc;
            }
            QPushButton:hover {
                background-color: #284e80;
            }
        """)
        btn_browse_lib.clicked.connect(self.sample_library_requested.emit)
        sam_layout.addWidget(btn_browse_lib)

        self.sample_list = QListWidget()
        self.sample_list.setStyleSheet("""
            QListWidget {
                background-color: #121214;
                color: #e0e0e0;
                border: 1px solid #282828;
                border-radius: 3px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 8px 6px;
                border-bottom: 1px solid #1c1c1f;
            }
            QListWidget::item:hover {
                background-color: #26262a;
            }
            QListWidget::item:selected {
                background-color: #005999;
                color: #ffffff;
            }
        """)
        self.sample_list.itemDoubleClicked.connect(self._on_sample_double_clicked)
        sam_layout.addWidget(self.sample_list, stretch=1)

        self.btn_open_sample = QPushButton("Open Selected Case")
        self.btn_open_sample.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2e;
                color: #e0e0e0;
                font-weight: bold;
                padding: 5px;
                border: 1px solid #3d3d40;
                border-radius: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #007acc;
                color: white;
            }
        """)
        self.btn_open_sample.clicked.connect(self._on_open_sample_clicked)
        sam_layout.addWidget(self.btn_open_sample)

        self.tabs.addTab(sample_page, "Sample Cases")
        layout.addWidget(self.tabs)

        self.refresh_sample_list()

    def refresh_sample_list(self):
        self.sample_list.clear()
        cases = self.sample_mgr.get_cases()
        for c in cases:
            status_tag = "Ready" if c.is_downloaded else "Not Downloaded"
            item = QListWidgetItem(f"[{c.modality}] {c.title}\n{c.anatomical_region} ({status_tag})")
            item.setData(Qt.UserRole, c.id)
            self.sample_list.addItem(item)

    def set_series_map(self, series_map: Dict[str, DicomSeriesGroup]):
        self.tree.clear()
        self.btn_load_series.setEnabled(False)

        # Group by study
        studies: Dict[str, list] = {}
        for s in series_map.values():
            studies.setdefault(s.study_uid, []).append(s)

        for study_uid, s_list in studies.items():
            first = s_list[0]
            p_name = first.patient_name.replace("^", " ")
            study_item = QTreeWidgetItem([
                f"{p_name} ({first.patient_id}) - {first.study_description}",
                "",
                f"{len(s_list)} series"
            ])
            study_item.setExpanded(True)
            study_item.setForeground(0, Qt.cyan)

            for s in s_list:
                s_item = QTreeWidgetItem([
                    s.series_description,
                    s.modality,
                    str(s.slice_count)
                ])
                s_item.setData(0, Qt.UserRole, s.series_uid)
                study_item.addChild(s_item)

            self.tree.addTopLevelItem(study_item)

        self.tabs.setCurrentIndex(0)

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        series_uid = item.data(0, Qt.UserRole)
        if series_uid:
            self.series_selected.emit(series_uid)

    def _on_load_selected_clicked(self):
        item = self.tree.currentItem()
        if item:
            series_uid = item.data(0, Qt.UserRole)
            if series_uid:
                self.series_selected.emit(series_uid)

    def _on_sample_double_clicked(self, item: QListWidgetItem):
        case_id = item.data(Qt.UserRole)
        if case_id:
            self.sample_case_selected.emit(case_id)

    def _on_open_sample_clicked(self):
        item = self.sample_list.currentItem()
        if item:
            case_id = item.data(Qt.UserRole)
            if case_id:
                self.sample_case_selected.emit(case_id)
