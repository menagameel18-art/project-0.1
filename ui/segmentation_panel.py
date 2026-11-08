"""
Segmentation Editor Control Panel.
Provides user interface for multi-segment management, color selection,
manual brush/contour drawing, and semi-automatic thresholding/region growing algorithms.
"""

from __future__ import annotations
import os
from typing import Optional, Dict
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton,
    QSlider, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QColorDialog, QMessageBox, QFileDialog, QTabWidget,
    QGroupBox, QRadioButton, QButtonGroup, QProgressBar, QComboBox,
    QCheckBox, QSplitter
)

from segmentation.manager import manager, SegmentationManager
from imaging.volume_processor import VolumeProcessor

class SegmentationPanel(QWidget):
    """Interactive sidebar panel for segmentation workflow."""
    tool_requested = Signal(str)  # tool_id
    segments_updated = Signal()

    def __init__(self, segmentation_manager: SegmentationManager, parent=None):
        super().__init__(parent)
        self.seg_manager = segmentation_manager
        self.processor: Optional[VolumeProcessor] = None

        self._setup_ui()
        self._connect_signals()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_processor(self, processor: VolumeProcessor):
        self.processor = processor
        self._populate_intensity_ranges()
        self.refresh_segment_table()

    # ---------------------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------------------
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Panel Header Bar
        header_widget = QWidget()
        h_layout = QHBoxLayout(header_widget)
        h_layout.setContentsMargins(2, 2, 2, 2)
        h_layout.setSpacing(4)

        title = QLabel("SEGMENTATION EDITOR")
        title.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(title)
        h_layout.addStretch()

        # Header buttons: Save, Load, Run AI Segmentation
        self.btn_save = self._create_mini_btn(
            "Save...", "Save segmentation project (.seg.npz)", self._on_save_clicked)
        self.btn_load = self._create_mini_btn(
            "Load...", "Load segmentation project (.seg.npz)", self._on_load_clicked)
        self.btn_ai = self._create_mini_btn(
            "Run AI Seg", "Run TotalSegmentator segmentation on current volume", self._on_run_ai_clicked)
        h_layout.addWidget(self.btn_save)
        h_layout.addWidget(self.btn_load)
        h_layout.addWidget(self.btn_ai)
        main_layout.addWidget(header_widget)

        # 2. Segment List Section
        seg_box = QGroupBox("Segments")
        seg_box.setStyleSheet("""
            QGroupBox {
                color: #aaaaaa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #2d2d34;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        seg_box_layout = QVBoxLayout(seg_box)
        seg_box_layout.setContentsMargins(6, 6, 6, 6)
        seg_box_layout.setSpacing(6)

        # Table of segments
        self.table_segments = QTableWidget(0, 4)
        self.table_segments.setHorizontalHeaderLabels(["Vis", "Color", "Name", "Vol (cm³)"])
        self.table_segments.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_segments.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_segments.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_segments.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_segments.verticalHeader().setVisible(False)
        self.table_segments.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_segments.setSelectionMode(QTableWidget.SingleSelection)
        self.table_segments.setFixedHeight(120)
        self.table_segments.setStyleSheet("""
            QTableWidget {
                background-color: #17171a;
                border: 1px solid #26262c;
                border-radius: 3px;
                color: #e0e0e0;
                font-size: 11px;
            }
            QTableWidget::item:selected {
                background-color: #0e3d59;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #202026;
                color: #999999;
                font-size: 10px;
                border: none;
                padding: 3px;
            }
        """)
        self.table_segments.itemSelectionChanged.connect(self._on_table_selection_changed)
        seg_box_layout.addWidget(self.table_segments)

        # Buttons under table: New, Delete, Clear, Undo, Redo
        seg_btn_row = QHBoxLayout()
        seg_btn_row.setSpacing(4)

        self.btn_add_seg = self._create_mini_btn("+ Add", "Add new independent segment", self._on_add_segment)
        self.btn_del_seg = self._create_mini_btn("Delete", "Delete selected segment (with confirmation)", self._on_delete_segment)
        self.btn_clear_seg = self._create_mini_btn("Clear", "Clear mask of selected segment", self._on_clear_segment)
        self.btn_undo = self._create_mini_btn("Undo", "Undo last segmentation stroke (Ctrl+Z)", self.seg_manager.undo)
        self.btn_redo = self._create_mini_btn("Redo", "Redo last segmentation stroke (Ctrl+Y)", self.seg_manager.redo)
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        seg_btn_row.addWidget(self.btn_add_seg)
        seg_btn_row.addWidget(self.btn_del_seg)
        seg_btn_row.addWidget(self.btn_clear_seg)
        seg_btn_row.addStretch()
        seg_btn_row.addWidget(self.btn_undo)
        seg_btn_row.addWidget(self.btn_redo)
        seg_box_layout.addLayout(seg_btn_row)

        # Global Opacity Slider
        op_layout = QHBoxLayout()
        op_layout.setContentsMargins(0, 2, 0, 0)
        lbl_op = QLabel("Overlay Opacity:")
        lbl_op.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        op_layout.addWidget(lbl_op)

        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(5, 100)
        self.slider_opacity.setValue(70)
        self.slider_opacity.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #333333; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #00d2ff; border-radius: 2px; }
            QSlider::handle:horizontal { background: #ffffff; width: 10px; margin-top: -3px; margin-bottom: -3px; border-radius: 5px; }
        """)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        op_layout.addWidget(self.slider_opacity)

        self.lbl_opacity_val = QLabel("70%")
        self.lbl_opacity_val.setStyleSheet("color: #00d2ff; font-size: 10px; font-weight: bold; min-width: 30px;")
        op_layout.addWidget(self.lbl_opacity_val)
        seg_box_layout.addLayout(op_layout)

        main_layout.addWidget(seg_box)

        # 3. Method Tabs: Manual vs Semi-Automatic
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2d2d34;
                background-color: #161619;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #202026;
                color: #aaaaaa;
                padding: 5px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #161619;
                color: #00d2ff;
                border-bottom: 2px solid #00d2ff;
            }
        """)

        # Tab 1: Manual Segmentation
        tab_manual = QWidget()
        manual_layout = QVBoxLayout(tab_manual)
        manual_layout.setContentsMargins(8, 8, 8, 8)
        manual_layout.setSpacing(8)

        # Tool selection radio group
        tool_group_box = QGroupBox("Drawing Tool")
        tool_group_box.setStyleSheet("color: #aaaaaa; font-size: 10px; border: 1px solid #2a2a30; border-radius: 3px; padding-top: 8px;")
        tool_layout = QHBoxLayout(tool_group_box)
        tool_layout.setSpacing(6)

        self.rb_none = QRadioButton("Navigate")
        self.rb_brush = QRadioButton("Brush")
        self.rb_eraser = QRadioButton("Eraser")
        self.rb_contour = QRadioButton("Contour")
        self.rb_none.setChecked(True)

        tool_btn_group = QButtonGroup(self)
        tool_btn_group.addButton(self.rb_none)
        tool_btn_group.addButton(self.rb_brush)
        tool_btn_group.addButton(self.rb_eraser)
        tool_btn_group.addButton(self.rb_contour)
        tool_btn_group.buttonClicked.connect(self._on_tool_radio_clicked)

        tool_layout.addWidget(self.rb_none)
        tool_layout.addWidget(self.rb_brush)
        tool_layout.addWidget(self.rb_eraser)
        tool_layout.addWidget(self.rb_contour)
        manual_layout.addWidget(tool_group_box)

        # Brush Size Slider
        brush_size_row = QHBoxLayout()
        lbl_bsize = QLabel("Brush Radius:")
        lbl_bsize.setStyleSheet("color: #cccccc; font-size: 11px;")
        brush_size_row.addWidget(lbl_bsize)

        self.slider_brush_size = QSlider(Qt.Horizontal)
        self.slider_brush_size.setRange(1, 40)
        self.slider_brush_size.setValue(6)
        self.slider_brush_size.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #333333; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #4ade80; border-radius: 2px; }
            QSlider::handle:horizontal { background: #ffffff; width: 10px; margin-top: -3px; margin-bottom: -3px; border-radius: 5px; }
        """)
        self.slider_brush_size.valueChanged.connect(self._on_brush_size_changed)
        brush_size_row.addWidget(self.slider_brush_size)

        self.lbl_brush_size_val = QLabel("6 vox")
        self.lbl_brush_size_val.setStyleSheet("color: #4ade80; font-size: 10px; font-weight: bold; min-width: 38px;")
        brush_size_row.addWidget(self.lbl_brush_size_val)
        manual_layout.addLayout(brush_size_row)

        # Contour action row
        contour_row = QHBoxLayout()
        self.btn_fill_contour = self._create_btn("Fill Contour", "Close active contour and fill region", self._on_fill_contour)
        self.btn_clear_contour = self._create_btn("Clear Points", "Clear active contour points", self._on_clear_contour)
        contour_row.addWidget(self.btn_fill_contour)
        contour_row.addWidget(self.btn_clear_contour)
        manual_layout.addLayout(contour_row)

        contour_hint = QLabel("Contour: Click vertices on any 2D slice. Double-click or click near start to fill.")
        contour_hint.setStyleSheet("color: #777788; font-size: 10px;")
        contour_hint.setWordWrap(True)
        manual_layout.addWidget(contour_hint)

        manual_layout.addStretch()
        self.tabs.addTab(tab_manual, "Manual")

        # Tab 2: Semi-Automatic Segmentation
        tab_semi = QWidget()
        semi_layout = QVBoxLayout(tab_semi)
        semi_layout.setContentsMargins(8, 8, 8, 8)
        semi_layout.setSpacing(6)

        # Algorithm selection
        algo_row = QHBoxLayout()
        lbl_algo = QLabel("Algorithm:")
        lbl_algo.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["Thresholding", "Region Growing"])
        self.combo_algo.setStyleSheet("""
            QComboBox { background-color: #24242c; color: #ffffff; border: 1px solid #3d3d48; border-radius: 3px; padding: 2px 6px; font-size: 11px; }
        """)
        self.combo_algo.currentIndexChanged.connect(self._on_algo_changed)
        algo_row.addWidget(lbl_algo)
        algo_row.addWidget(self.combo_algo, stretch=1)
        semi_layout.addLayout(algo_row)

        # Parameter Stack for Thresholding
        self.widget_threshold = QWidget()
        th_layout = QVBoxLayout(self.widget_threshold)
        th_layout.setContentsMargins(0, 4, 0, 4)
        th_layout.setSpacing(6)

        # Min threshold
        th_min_row = QHBoxLayout()
        self.lbl_th_min = QLabel("Min (HU):")
        self.lbl_th_min.setStyleSheet("color: #cccccc; font-size: 11px; min-width: 60px;")
        self.spin_th_min = QDoubleSpinBox()
        self.spin_th_min.setRange(-2000.0, 5000.0)
        self.spin_th_min.setValue(200.0)
        self.spin_th_min.setSingleStep(10.0)
        self.spin_th_min.setStyleSheet("background-color: #24242c; color: #ffffff; border: 1px solid #3d3d48;")
        th_min_row.addWidget(self.lbl_th_min)
        th_min_row.addWidget(self.spin_th_min, stretch=1)
        th_layout.addLayout(th_min_row)

        # Max threshold
        th_max_row = QHBoxLayout()
        self.lbl_th_max = QLabel("Max (HU):")
        self.lbl_th_max.setStyleSheet("color: #cccccc; font-size: 11px; min-width: 60px;")
        self.spin_th_max = QDoubleSpinBox()
        self.spin_th_max.setRange(-2000.0, 5000.0)
        self.spin_th_max.setValue(1200.0)
        self.spin_th_max.setSingleStep(10.0)
        self.spin_th_max.setStyleSheet("background-color: #24242c; color: #ffffff; border: 1px solid #3d3d48;")
        th_max_row.addWidget(self.lbl_th_max)
        th_max_row.addWidget(self.spin_th_max, stretch=1)
        th_layout.addLayout(th_max_row)

        # Threshold presets
        th_presets_row = QHBoxLayout()
        lbl_th_pre = QLabel("Preset:")
        lbl_th_pre.setStyleSheet("color: #888888; font-size: 10px;")
        self.combo_th_presets = QComboBox()
        self.combo_th_presets.addItems([
            "Custom",
            "Bone (> 200 HU)",
            "Soft Tissue (-50 to 150 HU)",
            "Lung (-900 to -500 HU)",
            "Fat (-150 to -40 HU)"
        ])
        self.combo_th_presets.setStyleSheet("background-color: #202026; color: #cccccc; font-size: 10px;")
        self.combo_th_presets.currentIndexChanged.connect(self._on_th_preset_changed)
        th_presets_row.addWidget(lbl_th_pre)
        th_presets_row.addWidget(self.combo_th_presets, stretch=1)
        th_layout.addLayout(th_presets_row)

        semi_layout.addWidget(self.widget_threshold)

        # Parameter Stack for Region Growing
        self.widget_rg = QWidget()
        self.widget_rg.setVisible(False)
        rg_layout = QVBoxLayout(self.widget_rg)
        rg_layout.setContentsMargins(0, 4, 0, 4)
        rg_layout.setSpacing(6)

        # Pick Seed Point toggle button
        self.btn_pick_seed = QPushButton("Pick Seed Point in Viewport")
        self.btn_pick_seed.setCheckable(True)
        self.btn_pick_seed.setStyleSheet("""
            QPushButton { background-color: #1e3a5f; color: #00d2ff; border: 1px solid #007acc; border-radius: 3px; padding: 4px; font-weight: bold; font-size: 11px; }
            QPushButton:checked { background-color: #007acc; color: #ffffff; }
        """)
        self.btn_pick_seed.toggled.connect(self._on_pick_seed_toggled)
        rg_layout.addWidget(self.btn_pick_seed)

        self.lbl_seed_status = QLabel("Seed: None selected")
        self.lbl_seed_status.setStyleSheet("color: #eab308; font-size: 10px; font-weight: bold;")
        rg_layout.addWidget(self.lbl_seed_status)

        # Tolerance
        tol_row = QHBoxLayout()
        lbl_tol = QLabel("Tolerance (+/-):")
        lbl_tol.setStyleSheet("color: #cccccc; font-size: 11px;")
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1.0, 500.0)
        self.spin_tol.setValue(35.0)
        self.spin_tol.setSingleStep(5.0)
        self.spin_tol.setStyleSheet("background-color: #24242c; color: #ffffff; border: 1px solid #3d3d48;")
        tol_row.addWidget(lbl_tol)
        tol_row.addWidget(self.spin_tol, stretch=1)
        rg_layout.addLayout(tol_row)

        # Connectivity
        conn_row = QHBoxLayout()
        lbl_conn = QLabel("Connectivity:")
        lbl_conn.setStyleSheet("color: #cccccc; font-size: 11px;")
        self.combo_conn = QComboBox()
        self.combo_conn.addItems(["6-Connected (Orthogonal)", "26-Connected (All directions)"])
        self.combo_conn.setStyleSheet("background-color: #202026; color: #cccccc; font-size: 10px;")
        conn_row.addWidget(lbl_conn)
        conn_row.addWidget(self.combo_conn, stretch=1)
        rg_layout.addLayout(conn_row)

        semi_layout.addWidget(self.widget_rg)

        # Execution Scope
        self.chk_slice_only = QCheckBox("Active 2D slice only (uncheck for full 3D volume)")
        self.chk_slice_only.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        semi_layout.addWidget(self.chk_slice_only)

        # Preview Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background-color: #202026; border-radius: 4px; }
            QProgressBar::chunk { background-color: #00d2ff; border-radius: 4px; }
        """)
        semi_layout.addWidget(self.progress_bar)

        self.lbl_worker_status = QLabel("")
        self.lbl_worker_status.setStyleSheet("color: #00d2ff; font-size: 10px;")
        semi_layout.addWidget(self.lbl_worker_status)

        # Semi-auto Action Buttons: Preview, Apply, Cancel
        action_row = QHBoxLayout()
        self.btn_preview = self._create_btn("Preview", "Calculate and preview segmentation result", self._on_preview_clicked)
        self.btn_preview.setStyleSheet("""
            QPushButton { background-color: #78350f; color: #fde047; border: 1px solid #b45309; border-radius: 3px; font-weight: bold; padding: 4px; font-size: 11px; }
            QPushButton:hover { background-color: #92400e; }
        """)
        self.btn_apply = self._create_btn("Apply", "Commit preview result to selected segment", self._on_apply_clicked)
        self.btn_apply.setEnabled(False)
        self.btn_apply.setStyleSheet("""
            QPushButton { background-color: #14532d; color: #86efac; border: 1px solid #16a34a; border-radius: 3px; font-weight: bold; padding: 4px; font-size: 11px; }
            QPushButton:hover { background-color: #166534; }
            QPushButton:disabled { background-color: #202024; color: #555555; border-color: #333333; }
        """)
        self.btn_cancel_preview = self._create_btn("Cancel", "Clear preview", self._on_cancel_preview_clicked)
        action_row.addWidget(self.btn_preview)
        action_row.addWidget(self.btn_apply)
        action_row.addWidget(self.btn_cancel_preview)
        semi_layout.addLayout(action_row)

        semi_layout.addStretch()
        self.tabs.addTab(tab_semi, "Semi-Auto")

        main_layout.addWidget(self.tabs, stretch=1)

    # ---------------------------------------------------------------------
    # Signal Connections
    # ---------------------------------------------------------------------
    def _connect_signals(self):
        self.seg_manager.segment_list_changed.connect(self.refresh_segment_table)
        self.seg_manager.segment_updated.connect(lambda _: self._update_segment_volume_display())
        self.seg_manager.active_segment_changed.connect(self._select_table_row)
        self.seg_manager.undo_redo_state_changed.connect(self._update_undo_redo_buttons)
        self.seg_manager.preview_updated.connect(self._on_preview_updated)
        self.seg_manager.worker_progress.connect(self._on_worker_progress)
        self.seg_manager.worker_finished.connect(self._on_worker_finished)
        self.seg_manager.worker_error.connect(self._on_worker_error)

    # ---------------------------------------------------------------------
    # Helper methods & UI callbacks (unchanged apart from formatting)
    # ---------------------------------------------------------------------
    def _populate_intensity_ranges(self):
        if not self.processor or self.processor.volume is None:
            return
        is_ct = self.processor.metadata.is_ct
        unit = "HU" if is_ct else "Val"
        self.lbl_th_min.setText(f"Min ({unit}):")
        self.lbl_th_max.setText(f"Max ({unit}):")

        v_min = float(np.percentile(self.processor.volume, 1))
        v_max = float(np.percentile(self.processor.volume, 99))

        self.spin_th_min.setRange(v_min - 1000, v_max + 1000)
        self.spin_th_max.setRange(v_min - 1000, v_max + 1000)

        if is_ct:
            self.spin_th_min.setValue(200.0)
            self.spin_th_max.setValue(1200.0)
        else:
            self.spin_th_min.setValue(v_min + (v_max - v_min) * 0.4)
            self.spin_th_max.setValue(v_max)

    def refresh_segment_table(self):
        if not self.seg_manager.model:
            self.table_segments.setRowCount(0)
            return

        segments = list(self.seg_manager.model.segments.values())
        self.table_segments.blockSignals(True)
        self.table_segments.setRowCount(len(segments))

        for row, seg in enumerate(segments):
            # Visibility checkbox
            chk_vis = QCheckBox()
            chk_vis.setChecked(seg.visible)
            chk_vis.setStyleSheet("margin-left: 6px;")
            chk_vis.toggled.connect(lambda state, s_id=seg.id: self.seg_manager.set_segment_visibility(s_id, state))
            self.table_segments.setCellWidget(row, 0, chk_vis)

            # Color swatch button
            btn_color = QPushButton()
            r, g, b, _ = seg.color
            btn_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #ffffff; border-radius: 2px;")
            btn_color.setFixedSize(16, 16)
            btn_color.clicked.connect(lambda _, s_id=seg.id: self._pick_color_for_segment(s_id))
            swatch_container = QWidget()
            sw_layout = QHBoxLayout(swatch_container)
            sw_layout.setContentsMargins(4, 2, 4, 2)
            sw_layout.addWidget(btn_color)
            self.table_segments.setCellWidget(row, 1, swatch_container)

            # Name item
            item_name = QTableWidgetItem(seg.name)
            item_name.setData(Qt.UserRole, seg.id)
            self.table_segments.setItem(row, 2, item_name)

            # Volume item
            geom = self.seg_manager.model.geometry
            vol_cm3 = seg.calculate_volume_cm3(geom)
            item_vol = QTableWidgetItem(f"{vol_cm3:.1f}")
            item_vol.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table_segments.setItem(row, 3, item_vol)

        self.table_segments.blockSignals(False)
        self._select_table_row(self.seg_manager.active_segment_id)

    def _select_table_row(self, segment_id: Optional[int]):
        if segment_id is None:
            return
        for r in range(self.table_segments.rowCount()):
            item = self.table_segments.item(r, 2)
            if item and item.data(Qt.UserRole) == segment_id:
                self.table_segments.blockSignals(True)
                self.table_segments.selectRow(r)
                self.table_segments.blockSignals(False)
                break

    def _update_segment_volume_display(self):
        if not self.seg_manager.model:
            return
        geom = self.seg_manager.model.geometry
        for r in range(self.table_segments.rowCount()):
            item_name = self.table_segments.item(r, 2)
            if item_name:
                s_id = item_name.data(Qt.UserRole)
                seg = self.seg_manager.model.get_segment(s_id)
                if seg:
                    vol_cm3 = seg.calculate_volume_cm3(geom)
                    item_vol = self.table_segments.item(r, 3)
                    if item_vol:
                        item_vol.setText(f"{vol_cm3:.1f}")

    def _on_table_selection_changed(self):
        sel_rows = self.table_segments.selectedIndexes()
        if not sel_rows:
            return
        row = sel_rows[0].row()
        item = self.table_segments.item(row, 2)
        if item:
            s_id = item.data(Qt.UserRole)
            self.seg_manager.set_active_segment(s_id)

    def _pick_color_for_segment(self, segment_id: int):
        seg = self.seg_manager.model.get_segment(segment_id) if self.seg_manager.model else None
        if not seg:
            return
        curr_r, curr_g, curr_b, curr_a = seg.color
        color = QColorDialog.getColor(QColor(curr_r, curr_g, curr_b, curr_a), self, f"Select Color for {seg.name}")
        if color.isValid():
            new_rgba = (color.red(), color.green(), color.blue(), 200)
            self.seg_manager.set_segment_color(segment_id, new_rgba)
            self.refresh_segment_table()

    def _on_add_segment(self):
        self.seg_manager.add_segment()

    def _on_delete_segment(self):
        seg = self.seg_manager.get_active_segment()
        if not seg:
            return
        reply = QMessageBox.question(
            self,
            "Confirm Delete Segment",
            f"Are you sure you want to permanently delete '{seg.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.seg_manager.remove_segment(seg.id)

    def _on_clear_segment(self):
        seg = self.seg_manager.get_active_segment()
        if not seg:
            return
        reply = QMessageBox.question(
            self,
            "Confirm Clear Segment",
            f"Are you sure you want to clear all voxels from '{seg.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.seg_manager.clear_segment(seg.id)

    def _on_opacity_changed(self, val: int):
        self.lbl_opacity_val.setText(f"{val}%")
        self.seg_manager.set_global_opacity(val / 100.0)

    def _on_tool_radio_clicked(self, btn):
        if btn == self.rb_brush:
            self.tool_requested.emit("segment_brush")
        elif btn == self.rb_eraser:
            self.tool_requested.emit("segment_eraser")
        elif btn == self.rb_contour:
            self.tool_requested.emit("segment_contour")
        else:
            self.tool_requested.emit("window_level")

    def _on_brush_size_changed(self, val: int):
        self.lbl_brush_size_val.setText(f"{val} vox")
        self.seg_manager.brush_tool.set_radius(val)

    def _on_fill_contour(self):
        self.seg_manager.finish_contour()

    def _on_clear_contour(self):
        self.seg_manager.contour_tool.clear()
        self.seg_manager.preview_updated.emit()

    def _on_algo_changed(self, idx: int):
        is_threshold = (idx == 0)
        self.widget_threshold.setVisible(is_threshold)
        self.widget_rg.setVisible(not is_threshold)
        if not is_threshold and self.btn_pick_seed.isChecked():
            self.tool_requested.emit("segment_seed_picker")
        else:
            if self.rb_brush.isChecked():
                self.tool_requested.emit("segment_brush")

    def _on_th_preset_changed(self, idx: int):
        txt = self.combo_th_presets.currentText()
        if "Bone" in txt:
            self.spin_th_min.setValue(200.0)
            self.spin_th_max.setValue(1500.0)
        elif "Soft Tissue" in txt:
            self.spin_th_min.setValue(-50.0)
            self.spin_th_max.setValue(150.0)
        elif "Lung" in txt:
            self.spin_th_min.setValue(-900.0)
            self.spin_th_max.setValue(-500.0)
        elif "Fat" in txt:
            self.spin_th_min.setValue(-150.0)
            self.spin_th_max.setValue(-40.0)

    def _on_pick_seed_toggled(self, checked: bool):
        if checked:
            self.tool_requested.emit("segment_seed_picker")
        else:
            if self.rb_brush.isChecked():
                self.tool_requested.emit("segment_brush")
            else:
                self.tool_requested.emit("window_level")

    def _on_preview_clicked(self):
        if not self.processor or self.processor.volume is None:
            QMessageBox.information(self, "No Volume", "Please load a medical study first.")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.lbl_worker_status.setText("Processing algorithm...")

        is_threshold = (self.combo_algo.currentIndex() == 0)
        slice_only = self.chk_slice_only.isChecked()

        if is_threshold:
            min_v = self.spin_th_min.value()
            max_v = self.spin_th_max.value()
            self.seg_manager.start_threshold_preview(
                volume=self.processor.volume,
                min_val=min_v,
                max_val=max_v,
                slice_only=slice_only
            )
        else:
            if not self.seg_manager.seed_point:
                QMessageBox.warning(self, "Seed Required", "Please click 'Pick Seed Point' and click on a slice to pick a seed voxel.")
                self.progress_bar.setVisible(False)
                self.lbl_worker_status.setText("")
                return
            tol = self.spin_tol.value()
            conn = 26 if "26" in self.combo_conn.currentText() else 6
            self.seg_manager.start_region_growing_preview(
                volume=self.processor.volume,
                seed_point=self.seg_manager.seed_point,
                tolerance=tol,
                connectivity=conn,
                slice_only=slice_only
            )

    def _on_apply_clicked(self):
        self.seg_manager.apply_preview(combine_mode="union")
        self.btn_apply.setEnabled(False)

    def _on_cancel_preview_clicked(self):
        self.seg_manager.cancel_worker()
        self.seg_manager.clear_preview()
        self.progress_bar.setVisible(False)
        self.lbl_worker_status.setText("")
        self.btn_apply.setEnabled(False)

    def _on_preview_updated(self):
        has_preview = (self.seg_manager.preview_mask is not None and np.any(self.seg_manager.preview_mask > 0))
        self.btn_apply.setEnabled(has_preview)
        if self.seg_manager.seed_point:
            z, y, x = self.seg_manager.seed_point
            val_str = f" | {self.seg_manager.seed_intensity:.1f}" if self.seg_manager.seed_intensity is not None else ""
            self.lbl_seed_status.setText(f"Seed: ({z}, {y}, {x}){val_str}")

    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_worker_status.setText(msg)

    def _on_worker_finished(self):
        self.progress_bar.setVisible(False)
        self.lbl_worker_status.setText("Preview ready. Click 'Apply' to commit.")

    def _on_worker_error(self, err: str):
        self.progress_bar.setVisible(False)
        self.lbl_worker_status.setText(f"Error: {err}")
        QMessageBox.warning(self, "Algorithm Error", err)

    def _update_undo_redo_buttons(self, can_undo: bool, can_redo: bool):
        self.btn_undo.setEnabled(can_undo)
        self.btn_redo.setEnabled(can_redo)

    def _on_save_clicked(self):
        if not self.seg_manager.model:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation Project",
            os.path.expanduser("~/segmentation.seg.npz"),
            "Segmentation Files (*.seg.npz)"
        )
        if path:
            p_id = self.processor.metadata.patient_id if self.processor else "Unknown"
            s_uid = self.processor.metadata.series_uid if self.processor else ""
            if self.seg_manager.save_project(path, patient_id=p_id, series_uid=s_uid):
                QMessageBox.information(self, "Saved", f"Segmentation saved successfully to:\n{path}")

    def _on_load_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Segmentation Project",
            os.path.expanduser("~"),
            "Segmentation Files (*.seg.npz)"
        )
        if path:
            try:
                self.seg_manager.load_project(path)
                self.refresh_segment_table()
                QMessageBox.information(self, "Loaded", "Segmentation project loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _on_run_ai_clicked(self):
        """Handler for the "Run AI Seg" button.
        Executes TotalSegmentator on the current volume, registers the
        resulting masks with the global SegmentationManager, and refreshes the UI.
        """
        if not self.processor:
            QMessageBox.warning(self, "No volume", "Load a CT series before running AI segmentation.")
            return
        try:
            # 1️⃣ Export current volume to temporary NIfTI file
            import tempfile, pathlib, SimpleITK as sitk
            tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="ts_ai_"))
            nifti_path = tmp_dir / "input_volume.nii.gz"
            itk_img = sitk.GetImageFromArray(self.processor.volume)
            geom = self.processor.geometry
            itk_img.SetSpacing(geom.dx, geom.dy, geom.dz)
            itk_img.SetOrigin(geom.origin[0], geom.origin[1], geom.origin[2])
            itk_img.SetDirection(geom.direction)
            sitk.WriteImage(itk_img, str(nifti_path))

            # 2️⃣ Run TotalSegmentator
            from segmentation.totalsegmentator_adapter import run_totalsegmentator
            output_dir = tmp_dir / "seg_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            results = run_totalsegmentator(
                input_path=nifti_path,
                output_dir=output_dir,
                structures=None,  # None = all supported structures
                use_cuda=False,
            )

            # 3️⃣ Register each mask with the manager
            default_palette = [
                (255, 0, 0, 255),   # red
                (0, 255, 0, 255),   # green
                (0, 0, 255, 255),   # blue
                (255, 255, 0, 255), # yellow
                (255, 0, 255, 255), # magenta
                (0, 255, 255, 255), # cyan
            ]
            for idx, (struct_name, mask_path) in enumerate(results.items()):
                color = default_palette[idx % len(default_palette)]
                self.seg_manager.add_nifti_mask(
                    nifti_path=mask_path,
                    name=struct_name,
                    source="totalsegmentator",
                    color=color,
                )
            # 4️⃣ Notify UI to refresh segment list and 3D view
            self.refresh_segment_table()
            try:
                # Assuming parent has a volume_viewer attribute
                self.parent().volume_viewer.refresh_segments()
            except Exception:
                pass
            QMessageBox.information(self, "AI Segmentation", "TotalSegmentator finished and segments were added.")
        except Exception as exc:
            QMessageBox.critical(self, "AI Segmentation Error", str(exc))

    # ---------------------------------------------------------------------
    # UI button factories
    # ---------------------------------------------------------------------
    def _create_mini_btn(self, text: str, tooltip: str, callback) -> QPushButton:
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(22)
        btn.setStyleSheet("""
            QPushButton { background-color: #26262e; color: #d0d0d0; border: 1px solid #3d3d48; border-radius: 3px; font-size: 10px; padding: 2px 6px; }
            QPushButton:hover { background-color: #33333e; border-color: #007acc; }
        """)
        btn.clicked.connect(callback)
        return btn

    def _create_btn(self, text: str, tooltip: str, callback) -> QPushButton:
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(26)
        btn.setStyleSheet("""
            QPushButton { background-color: #2b2b34; color: #e0e0e0; border: 1px solid #40404c; border-radius: 3px; font-size: 11px; padding: 2px 8px; }
            QPushButton:hover { background-color: #353542; border-color: #007acc; }
        """)
        btn.clicked.connect(callback)
        return btn
