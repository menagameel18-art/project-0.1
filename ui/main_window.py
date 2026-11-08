"""
Main Radiology Workstation Application Window.
Integrates synchronized 2D viewports, VTK 3D volume rendering, study browser sidebar,
top clinical toolbar, and status bar.
"""
from __future__ import annotations
import os
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
    QTabWidget
)

from ui.toolbar import MainToolBar
from ui.status_bar import WorkstationStatusBar
from ui.study_browser import StudyBrowser
from ui.sample_library_dialog import SampleLibraryDialog
from ui.welcome_dialog import WelcomeDialog
from ui.metadata_dialog import MetadataDialog
from ui.segmentation_panel import SegmentationPanel
from viewers.viewport_container import ViewportContainer
from dicom.series_manager import SeriesManager
from dicom.sample_data_manager import SampleDataManager
from imaging.volume_processor import VolumeProcessor
from segmentation.manager import SegmentationManager


class MainWindow(QMainWindow):
    """Main workstation window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antigravity Medical DICOM 3D Workstation")
        self.resize(1380, 880)

        # Apply global dark radiology theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121214;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #e0e0e0;
            }
            QToolTip {
                background-color: #2b2b32;
                color: #ffffff;
                border: 1px solid #444450;
                padding: 4px;
            }
        """)

        # Core logic managers
        self.volume_processor = VolumeProcessor()
        self.series_manager = SeriesManager(self.volume_processor)
        self.sample_manager = SampleDataManager()
        self.segmentation_manager = SegmentationManager(self)

        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()

        # Show welcome dialog on first run after event loop starts
        QTimer.singleShot(150, self._show_welcome_if_needed)

    def _setup_ui(self):
        # Top Toolbar
        self.toolbar = MainToolBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # Status Bar
        self.status_bar = WorkstationStatusBar(self)
        self.setStatusBar(self.status_bar)

        # Central Splitter (Left Sidebar + Viewports Grid)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setStyleSheet("QSplitter::handle { background-color: #242428; width: 3px; }")

        # Sidebar with Tabs (Studies & Cases / Segmentation Editor)
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(260)
        self.sidebar_tabs.setMaximumWidth(460)
        self.sidebar_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #282830;
                background-color: #141417;
            }
            QTabBar::tab {
                background-color: #1c1c22;
                color: #888899;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #141417;
                color: #00d2ff;
                border-bottom: 2px solid #00d2ff;
            }
        """)

        # Tab 1: Study Browser Sidebar
        self.study_browser = StudyBrowser(self.sample_manager)
        self.sidebar_tabs.addTab(self.study_browser, "Studies")

        # Tab 2: Segmentation Editor Sidebar
        self.segmentation_panel = SegmentationPanel(self.segmentation_manager, self)
        self.sidebar_tabs.addTab(self.segmentation_panel, "Segmentation")

        self.main_splitter.addWidget(self.sidebar_tabs)

        # 2x2 Viewports Grid Container
        self.viewports = ViewportContainer()
        self.viewports.set_segmentation_manager(self.segmentation_manager)
        self.main_splitter.addWidget(self.viewports)

        # Splitter initial sizes: Sidebar 300px, Viewports remainder
        self.main_splitter.setSizes([300, 1080])
        self.setCentralWidget(self.main_splitter)

    def _connect_signals(self):
        # Toolbar actions
        self.toolbar.open_folder_requested.connect(self.prompt_open_folder)
        self.toolbar.sample_cases_requested.connect(self.open_sample_library)
        self.toolbar.demo_case_requested.connect(self.load_demo_case)
        self.toolbar.segmentation_panel_requested.connect(self._show_segmentation_tab)
        self.toolbar.preset_selected.connect(self.viewports.apply_window_preset)
        self.toolbar.tool_changed.connect(self._on_segmentation_tool_selected)
        self.toolbar.crosshair_toggled.connect(self.viewports.toggle_crosshair_link)
        self.toolbar.reset_layout_requested.connect(self.viewports.reset_layout)
        self.toolbar.reset_views_requested.connect(self.viewports.reset_views)
        self.toolbar.info_requested.connect(self.show_metadata_dialog)

        # Study Browser actions
        self.study_browser.open_folder_requested.connect(self.prompt_open_folder)
        self.study_browser.sample_library_requested.connect(self.open_sample_library)
        self.study_browser.series_selected.connect(self.series_manager.load_series)
        self.study_browser.sample_case_selected.connect(self.load_sample_case)

        # Segmentation panel actions
        self.segmentation_panel.tool_requested.connect(self._on_segmentation_tool_selected)

        # Series Manager signals
        self.series_manager.loading_started.connect(lambda msg: self.status_bar.show_progress(0, 100, msg))
        self.series_manager.loading_progress.connect(self.status_bar.show_progress)
        self.series_manager.series_map_updated.connect(self.study_browser.set_series_map)
        self.series_manager.volume_loaded.connect(self._on_volume_loaded)
        self.series_manager.loading_failed.connect(self._on_loading_failed)

        # Inspector updates
        self.viewports.pixel_inspected.connect(self.status_bar.update_inspector)

    def _setup_shortcuts(self):
        # Shortcuts for common radiology workflows
        QShortcut(QKeySequence("Ctrl+O"), self, self.prompt_open_folder)
        QShortcut(QKeySequence("Ctrl+L"), self, self.open_sample_library)
        QShortcut(QKeySequence("Ctrl+D"), self, self.load_demo_case)
        QShortcut(QKeySequence("Ctrl+R"), self, self.viewports.reset_views)
        QShortcut(QKeySequence("W"), self, lambda: self._on_segmentation_tool_selected("window_level"))
        QShortcut(QKeySequence("P"), self, lambda: self._on_segmentation_tool_selected("pan"))
        QShortcut(QKeySequence("Z"), self, lambda: self._on_segmentation_tool_selected("zoom"))
        QShortcut(QKeySequence("M"), self, lambda: self._on_segmentation_tool_selected("measure"))
        QShortcut(QKeySequence("C"), self, lambda: self._on_segmentation_tool_selected("crosshair"))
        QShortcut(QKeySequence("B"), self, lambda: self._on_segmentation_tool_selected("segment_brush"))
        QShortcut(QKeySequence("E"), self, lambda: self._on_segmentation_tool_selected("segment_eraser"))
        QShortcut(QKeySequence("Ctrl+Z"), self, self.segmentation_manager.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.segmentation_manager.redo)

    def _show_segmentation_tab(self):
        self.sidebar_tabs.setCurrentWidget(self.segmentation_panel)

    def _on_segmentation_tool_selected(self, tool_id: str):
        self.viewports.set_active_tool(tool_id)
        if tool_id == "segment_brush":
            self.toolbar.btn_brush_tool.setChecked(True)
            self.segmentation_panel.rb_brush.setChecked(True)
        elif tool_id == "segment_eraser":
            self.toolbar.btn_eraser_tool.setChecked(True)
            self.segmentation_panel.rb_eraser.setChecked(True)
        elif tool_id == "segment_contour":
            self.toolbar.btn_contour_tool.setChecked(True)
            self.segmentation_panel.rb_contour.setChecked(True)
        elif tool_id == "segment_seed_picker":
            self.segmentation_panel.btn_pick_seed.setChecked(True)
        elif tool_id == "window_level":
            self.toolbar.btn_wl_tool.setChecked(True)
            self.segmentation_panel.rb_none.setChecked(True)
        elif tool_id == "pan":
            self.toolbar.btn_pan_tool.setChecked(True)
            self.segmentation_panel.rb_none.setChecked(True)
        elif tool_id == "zoom":
            self.toolbar.btn_zoom_tool.setChecked(True)
            self.segmentation_panel.rb_none.setChecked(True)
        elif tool_id == "measure":
            self.toolbar.btn_measure_tool.setChecked(True)
            self.segmentation_panel.rb_none.setChecked(True)
        elif tool_id == "crosshair":
            self.toolbar.btn_crosshair_tool.setChecked(True)
            self.segmentation_panel.rb_none.setChecked(True)

    def _show_welcome_if_needed(self):
        if not self.volume_processor.has_data():
            welcome = WelcomeDialog(self)
            welcome.open_folder_requested.connect(self.prompt_open_folder)
            welcome.explore_samples_requested.connect(self.open_sample_library)
            welcome.load_demo_requested.connect(self.load_demo_case)
            welcome.exec()

    def prompt_open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing DICOM Files",
            os.path.expanduser("~")
        )
        if folder:
            self.series_manager.scan_directory(folder)

    def open_sample_library(self):
        dialog = SampleLibraryDialog(self.sample_manager, self)
        dialog.case_opened.connect(self.load_sample_case)
        dialog.exec()
        self.study_browser.refresh_sample_list()

    def load_demo_case(self):
        """Loads instant procedural CT Head 3D reconstruction."""
        self.status_bar.show_progress(0, 100, "Generating high-resolution 3D CT Head Demonstration...")
        try:
            demo_proc = self.sample_manager.load_case_processor("demo_head_ct")
            self.series_manager.load_direct_processor(demo_proc)
        except Exception as e:
            self._on_loading_failed(str(e))

    def load_sample_case(self, case_id: str):
        self.status_bar.show_progress(0, 100, f"Loading sample study ({case_id})...")
        try:
            case_proc = self.sample_manager.load_case_processor(case_id)
            self.series_manager.load_direct_processor(case_proc)
        except Exception as e:
            self._on_loading_failed(str(e))

    def _on_volume_loaded(self, processor: VolumeProcessor):
        self.volume_processor = processor
        self.viewports.set_processor(processor)
        self.segmentation_panel.set_processor(processor)
        self.status_bar.hide_progress(msg="Ready")
        self.status_bar.set_metadata(processor.metadata, processor.geometry)
        self.setWindowTitle(f"Antigravity Medical Workstation - {processor.metadata.patient_name} [{processor.metadata.modality}]")

    def _on_loading_failed(self, error_msg: str):
        self.status_bar.hide_progress(msg="Error")
        QMessageBox.warning(self, "Loading Notice", error_msg)

    def show_metadata_dialog(self):
        if not self.volume_processor or not self.volume_processor.has_data():
            QMessageBox.information(self, "No Study Loaded", "Please load a DICOM study or sample case first.")
            return
        diag = MetadataDialog(self.volume_processor, self)
        diag.exec()
