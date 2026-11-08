"""
Three-Dimensional Volume Rendering and Multi-Segment Surface Viewport using VTK.
Provides interactive GPU/CPU ray-casting, clinical color/opacity transfer presets,
independent volume & segmentation surface rendering, distinct 3D display modes,
per-segment visibility/color/opacity controls, and anatomical orientation markers.
"""
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton,
    QComboBox, QSlider, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QColorDialog, QFrame
)

from imaging.volume_processor import VolumeProcessor
from segmentation.surface import SegmentSurfaceExtractor


class Volume3DViewer(QWidget):
    """
    VTK 3D Medical Viewport supporting:
    - Mode A: Original Volume Rendering (CT/MRI presets, HU-aware, opacity/brightness/contrast controls)
    - Mode B: Segmentation Surface Rendering (per-segment 3D meshes aligned in patient coordinates)
    - Mode 1: Show Selected Segment Only
    - Mode 2: Show All Segments
    - Mode 3: Original Volume + Selected/All Segments (with volume transparency control)
    - Individual segment 3D visibility, color swatch, opacity controls
    - Quick actions: Show All, Hide All, Selected Only, Show/Hide Volume, Reset 3D View
    """
    activated = Signal(str)
    display_mode_changed = Signal(str)

    # 3D Display Mode Constants
    MODE_COMBINED = "combined"          # Mode 3: Original Volume + Selected/All Segments
    MODE_ALL_SEGMENTS = "all_segments"  # Mode 2: Show All Segments (Volume Hidden)
    MODE_SELECTED_ONLY = "selected_only"# Mode 1: Show Selected Segment Only
    MODE_VOLUME_ONLY = "volume_only"    # Mode A: Original Volume Rendering Only

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor: Optional[VolumeProcessor] = None
        self.is_active = False

        # VTK Rendering Pipeline components
        self.renderer: Optional[vtk.vtkRenderer] = None
        self.volume_mapper: Optional[vtk.vtkSmartVolumeMapper] = None
        self.volume_property: Optional[vtk.vtkVolumeProperty] = None
        self.volume_actor: Optional[vtk.vtkVolume] = None
        self.color_tf: Optional[vtk.vtkColorTransferFunction] = None
        self.opacity_tf: Optional[vtk.vtkPiecewiseFunction] = None
        self.axes_widget: Optional[vtk.vtkOrientationMarkerWidget] = None

        # Volume state
        self.volume_visible: bool = True
        self.volume_opacity_multiplier: float = 1.0  # [0.0 - 1.0]
        self._base_opacity_points: List[Tuple[float, float]] = []

        # 3D Display mode state
        self.display_mode: str = self.MODE_COMBINED
        self.smoothing_enabled: bool = True

        # 3D Segmentation surface management
        self.segmentation_manager = None
        self._segment_actors: Dict[int, vtk.vtkActor] = {}
        self._actor_voxel_counts: Dict[int, int] = {}
        self._segment_3d_visibility: Dict[int, bool] = {}  # segment_id -> bool

        self._setup_ui()
        self._init_vtk()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)

        # 1. Main Header Bar
        self.header_bar = QWidget()
        self.header_bar.setFixedHeight(30)
        self.header_bar.setStyleSheet("background-color: #1a1a1e; border-bottom: 1px solid #2d2d34;")
        h_layout = QHBoxLayout(self.header_bar)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(6)

        title_lbl = QLabel("3D RENDERING")
        title_lbl.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(title_lbl)

        # 3D Display Mode Dropdown
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Mode 3: Volume + Segments", self.MODE_COMBINED)
        self.combo_mode.addItem("Mode 2: Show All Segments", self.MODE_ALL_SEGMENTS)
        self.combo_mode.addItem("Mode 1: Show Selected Only", self.MODE_SELECTED_ONLY)
        self.combo_mode.addItem("Mode A: Volume Only", self.MODE_VOLUME_ONLY)
        self.combo_mode.setFixedHeight(22)
        self.combo_mode.setStyleSheet("""
            QComboBox {
                background-color: #24242c;
                color: #00d2ff;
                border: 1px solid #3d3d4a;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
                padding: 0 4px;
            }
        """)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_combo_changed)
        h_layout.addWidget(self.combo_mode)

        # Preset Dropdown
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["CT Bone", "CT Soft Tissue", "CT Lung", "MIP (Maximum Intensity)", "MR / Grayscale"])
        self.combo_preset.setFixedHeight(22)
        self.combo_preset.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 4px;
            }
        """)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        h_layout.addWidget(self.combo_preset)

        # Quality Dropdown
        self.combo_quality = QComboBox()
        self.combo_quality.addItems(["Normal Quality", "High Quality", "Fast / Draft"])
        self.combo_quality.setFixedHeight(22)
        self.combo_quality.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 4px;
            }
        """)
        self.combo_quality.currentIndexChanged.connect(self._on_quality_changed)
        h_layout.addWidget(self.combo_quality)

        # Volume Visibility Toggle Button
        self.btn_toggle_vol = QToolButton()
        self.btn_toggle_vol.setText("Vol: ON")
        self.btn_toggle_vol.setToolTip("Toggle Original Volume Rendering independently")
        self.btn_toggle_vol.setFixedHeight(22)
        self.btn_toggle_vol.setStyleSheet("""
            QToolButton {
                background-color: #1e3a5f;
                color: #00d2ff;
                border: 1px solid #007acc;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
                padding: 0 6px;
            }
            QToolButton:hover { background-color: #285282; }
        """)
        self.btn_toggle_vol.clicked.connect(self.toggle_volume)
        h_layout.addWidget(self.btn_toggle_vol)

        # Toggle 3D Display Controls Panel
        self.btn_toggle_panel = QToolButton()
        self.btn_toggle_panel.setText("3D Controls ▾")
        self.btn_toggle_panel.setToolTip("Show / Hide 3D Display & Visibility Panel")
        self.btn_toggle_panel.setFixedHeight(22)
        self.btn_toggle_panel.setStyleSheet("""
            QToolButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 6px;
            }
            QToolButton:hover { background-color: #383838; }
        """)
        self.btn_toggle_panel.clicked.connect(self._toggle_3d_panel_visible)
        h_layout.addWidget(self.btn_toggle_panel)

        # Reset Camera Button
        self.btn_reset_cam = QToolButton()
        self.btn_reset_cam.setText("Reset Cam")
        self.btn_reset_cam.setToolTip("Reset 3D Camera (Reset 3D View)")
        self.btn_reset_cam.setFixedHeight(22)
        self.btn_reset_cam.setStyleSheet("""
            QToolButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 6px;
            }
            QToolButton:hover {
                background-color: #383838;
                border-color: #007acc;
            }
        """)
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        h_layout.addWidget(self.btn_reset_cam)

        layout.addWidget(self.header_bar)

        # 2. Dedicated 3D Display & Individual Segment Visibility Panel (Expandable)
        self.panel_3d_controls = QWidget()
        self.panel_3d_controls.setStyleSheet("background-color: #151518; border-bottom: 1px solid #2a2a30;")
        p_layout = QVBoxLayout(self.panel_3d_controls)
        p_layout.setContentsMargins(6, 4, 6, 6)
        p_layout.setSpacing(4)

        # Row 1: Convenient Action Buttons & Surface Smoothing Checkbox
        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)

        self.btn_act_show_all = self._create_mini_btn("Show All Segments", "Show all 3D segment surfaces together", self.show_all_segments)
        self.btn_act_hide_all = self._create_mini_btn("Hide All Segments", "Hide all 3D segment surfaces", self.hide_all_segments)
        self.btn_act_selected_only = self._create_mini_btn("Show Selected Only", "Display selected segment alone", self.show_selected_only)
        self.btn_act_show_vol = self._create_mini_btn("Show Volume", "Make original volume visible", self.show_volume)
        self.btn_act_hide_vol = self._create_mini_btn("Hide Volume", "Hide original volume", self.hide_volume)
        self.btn_act_reset_3d = self._create_mini_btn("Reset 3D View", "Reset camera view", self.reset_3d_view)

        actions_row.addWidget(self.btn_act_show_all)
        actions_row.addWidget(self.btn_act_hide_all)
        actions_row.addWidget(self.btn_act_selected_only)
        actions_row.addWidget(self.btn_act_show_vol)
        actions_row.addWidget(self.btn_act_hide_vol)
        actions_row.addWidget(self.btn_act_reset_3d)
        actions_row.addStretch()

        self.chk_smooth = QCheckBox("Smooth Surfaces")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.setToolTip("Apply surface mesh smoothing for 3D rendering (does not modify mask data)")
        self.chk_smooth.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.chk_smooth.toggled.connect(self.set_smoothing_enabled)
        actions_row.addWidget(self.chk_smooth)

        p_layout.addLayout(actions_row)

        # Row 2: Volume Opacity / Transparency Control (Inspect internal structures)
        vol_op_row = QHBoxLayout()
        vol_op_row.setSpacing(6)
        lbl_vol_op = QLabel("Volume Transparency (Inspect Internal Structures):")
        lbl_vol_op.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        vol_op_row.addWidget(lbl_vol_op)

        self.slider_vol_opacity = QSlider(Qt.Horizontal)
        self.slider_vol_opacity.setRange(0, 100)
        self.slider_vol_opacity.setValue(100)
        self.slider_vol_opacity.setToolTip("Adjust original volume opacity to inspect internal structures and segments")
        self.slider_vol_opacity.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #333333; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #00d2ff; border-radius: 2px; }
            QSlider::handle:horizontal { background: #ffffff; width: 10px; margin-top: -3px; margin-bottom: -3px; border-radius: 5px; }
        """)
        self.slider_vol_opacity.valueChanged.connect(self._on_vol_opacity_slider_changed)
        vol_op_row.addWidget(self.slider_vol_opacity)

        self.lbl_vol_opacity_val = QLabel("100%")
        self.lbl_vol_opacity_val.setStyleSheet("color: #00d2ff; font-size: 10px; font-weight: bold; min-width: 32px;")
        vol_op_row.addWidget(self.lbl_vol_opacity_val)

        p_layout.addLayout(vol_op_row)

        # Row 3: Individual 3D Segments Table (Vis, Color, Opacity, Name)
        self.table_3d_segments = QTableWidget(0, 4)
        self.table_3d_segments.setHorizontalHeaderLabels(["3D Vis", "Color", "3D Opacity", "Segment Name"])
        self.table_3d_segments.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_3d_segments.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_3d_segments.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_3d_segments.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_3d_segments.verticalHeader().setVisible(False)
        self.table_3d_segments.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_3d_segments.setSelectionMode(QTableWidget.SingleSelection)
        self.table_3d_segments.setFixedHeight(110)
        self.table_3d_segments.setStyleSheet("""
            QTableWidget {
                background-color: #121215;
                border: 1px solid #24242a;
                border-radius: 3px;
                color: #e0e0e0;
                font-size: 10px;
            }
            QTableWidget::item:selected {
                background-color: #0e3d59;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1a1a20;
                color: #888899;
                font-size: 10px;
                border: none;
                padding: 2px 4px;
            }
        """)
        self.table_3d_segments.itemSelectionChanged.connect(self._on_table_row_selected)
        p_layout.addWidget(self.table_3d_segments)

        layout.addWidget(self.panel_3d_controls)

        # 3. VTK Widget Container
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vtkWidget.setStyleSheet("background-color: #0a0a0c;")
        layout.addWidget(self.vtkWidget, stretch=1)

        # 4. Bottom Info Bar
        self.footer_bar = QWidget()
        self.footer_bar.setFixedHeight(28)
        self.footer_bar.setStyleSheet("background-color: #161618; border-top: 1px solid #24242a;")
        f_layout = QHBoxLayout(self.footer_bar)
        f_layout.setContentsMargins(8, 0, 8, 0)
        f_layout.setSpacing(8)

        self.lbl_info = QLabel("Volume: No Data Loaded")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        f_layout.addWidget(self.lbl_info)

        f_layout.addStretch()

        hint_lbl = QLabel("Rotate: Left-Drag | Zoom: Right-Drag / Wheel | Pan: Shift+Left / Middle")
        hint_lbl.setStyleSheet("color: #666677; font-size: 10px;")
        f_layout.addWidget(hint_lbl)

        layout.addWidget(self.footer_bar)
        self.set_active(False)

    def _create_mini_btn(self, text: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(20)
        btn.setStyleSheet("""
            QToolButton {
                background-color: #24242c;
                color: #d0d0d8;
                border: 1px solid #383844;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 5px;
            }
            QToolButton:hover {
                background-color: #2f2f3c;
                border-color: #00d2ff;
                color: #ffffff;
            }
        """)
        btn.clicked.connect(callback)
        return btn

    def _toggle_3d_panel_visible(self):
        is_vis = not self.panel_3d_controls.isVisible()
        self.panel_3d_controls.setVisible(is_vis)
        self.btn_toggle_panel.setText("3D Controls ▾" if is_vis else "3D Controls ▸")

    def _init_vtk(self):
        """Initializes VTK renderer, volume mapper, transfer functions, and interactor."""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.04, 0.04, 0.06)  # Deep radiology navy-black
        self.renderer.SetBackground2(0.12, 0.12, 0.16)
        self.renderer.SetGradientBackground(True)

        ren_win = self.vtkWidget.GetRenderWindow()
        ren_win.AddRenderer(self.renderer)

        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        # Initialize mapper with dummy volume so input port 0 is always connected
        dummy = vtk.vtkImageData()
        dummy.SetDimensions(1, 1, 1)
        dummy.AllocateScalars(vtk.VTK_FLOAT, 1)
        self.volume_mapper.SetInputData(dummy)

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetAmbient(0.2)
        self.volume_property.SetDiffuse(0.8)
        self.volume_property.SetSpecular(0.25)
        self.volume_property.SetSpecularPower(15.0)

        self.color_tf = vtk.vtkColorTransferFunction()
        self.opacity_tf = vtk.vtkPiecewiseFunction()

        self.volume_property.SetColor(self.color_tf)
        self.volume_property.SetScalarOpacity(self.opacity_tf)

        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(self.volume_mapper)
        self.volume_actor.SetProperty(self.volume_property)
        self.renderer.AddVolume(self.volume_actor)

        # Standard trackball camera style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(style)
        self.vtkWidget.Initialize()

        # 3D Coordinate axes orientation marker
        try:
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(25, 25, 25)
            self.axes_widget = vtk.vtkOrientationMarkerWidget()
            self.axes_widget.SetOrientationMarker(axes)
            self.axes_widget.SetInteractor(self.vtkWidget)
            self.axes_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.InteractiveOff()
        except Exception:
            pass

    def set_processor(self, processor: VolumeProcessor):
        """Connects new volume data, updates VTK pipeline, and renders."""
        self.processor = processor
        if not processor or not processor.has_data():
            self.lbl_info.setText("Volume: No Data Loaded")
            return

        vtk_img = processor.get_vtk_image_data()
        if not vtk_img:
            return

        self.volume_mapper.SetInputData(vtk_img)

        # Apply appropriate initial preset based on modality
        if processor.metadata.is_ct:
            self.combo_preset.setCurrentText("CT Bone")
            self._apply_ct_bone_preset()
        else:
            self.combo_preset.setCurrentText("MR / Grayscale")
            self._apply_mr_preset()

        g = processor.geometry
        self.lbl_info.setText(
            f"Dimensions: {g.num_x}x{g.num_y}x{g.num_z} | Spacing: {g.dx:.2f}x{g.dy:.2f}x{g.dz:.2f} mm"
        )

        # Clear and rebuild segment surfaces for new volume geometry
        self._rebuild_all_segment_surfaces()
        self.refresh_3d_segments_table()
        self.reset_camera()

    def reset_camera(self):
        """Resets camera to show the complete volume centered in patient space."""
        if not self.renderer:
            return
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Elevation(15)
        cam.Azimuth(-20)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def reset_3d_view(self):
        """Convenience action: resets 3D camera to patient orientation."""
        self.reset_camera()

    # =========================================================================
    # 3D Display Modes & Volume Controls
    # =========================================================================

    def set_display_mode(self, mode: str):
        """
        Switches between:
        - MODE_COMBINED: Original Volume + Visible/Selected Segments
        - MODE_ALL_SEGMENTS: All visible segments together (Volume hidden)
        - MODE_SELECTED_ONLY: Only active segment in 3D (other segments hidden)
        - MODE_VOLUME_ONLY: Original volume only (all segment surfaces hidden)
        """
        self.display_mode = mode
        # Sync combo box without recursion
        self.combo_mode.blockSignals(True)
        idx = self.combo_mode.findData(mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)
        self.combo_mode.blockSignals(False)

        # If switching to combined mode and volume was off, turn volume on
        if mode == self.MODE_COMBINED and not self.volume_visible:
            self.volume_visible = True
            self.btn_toggle_vol.setText("Vol: ON")
            self.btn_toggle_vol.setStyleSheet("background-color: #1e3a5f; color: #00d2ff; border: 1px solid #007acc; border-radius: 3px; font-size: 10px; font-weight: bold; padding: 0 6px;")

        self._update_actor_visibilities()
        self.display_mode_changed.emit(mode)

    def _on_mode_combo_changed(self, idx: int):
        mode = self.combo_mode.currentData()
        if mode:
            self.set_display_mode(mode)

    def set_volume_visibility(self, visible: bool):
        """Toggles original volume rendering on or off independently."""
        self.volume_visible = visible
        if visible:
            self.btn_toggle_vol.setText("Vol: ON")
            self.btn_toggle_vol.setStyleSheet("background-color: #1e3a5f; color: #00d2ff; border: 1px solid #007acc; border-radius: 3px; font-size: 10px; font-weight: bold; padding: 0 6px;")
        else:
            self.btn_toggle_vol.setText("Vol: OFF")
            self.btn_toggle_vol.setStyleSheet("background-color: #2b2b2b; color: #888888; border: 1px solid #444444; border-radius: 3px; font-size: 10px; padding: 0 6px;")

        self._update_actor_visibilities()

    def toggle_volume(self):
        """Convenience action: toggles volume visibility."""
        self.set_volume_visibility(not self.volume_visible)

    def show_volume(self):
        """Convenience action: shows original volume rendering."""
        self.set_volume_visibility(True)
        if self.display_mode == self.MODE_ALL_SEGMENTS:
            self.set_display_mode(self.MODE_COMBINED)

    def hide_volume(self):
        """Convenience action: hides original volume rendering."""
        self.set_volume_visibility(False)

    def set_volume_opacity(self, factor: float):
        """Adjusts original volume opacity multiplier [0.0 - 1.0] to inspect internal structures."""
        self.volume_opacity_multiplier = max(0.0, min(1.0, float(factor)))
        pct = int(round(self.volume_opacity_multiplier * 100))
        self.slider_vol_opacity.blockSignals(True)
        self.slider_vol_opacity.setValue(pct)
        self.slider_vol_opacity.blockSignals(False)
        self.lbl_vol_opacity_val.setText(f"{pct}%")
        self._apply_volume_opacity_transfer()
        if self.renderer and self.vtkWidget:
            self.vtkWidget.GetRenderWindow().Render()

    def _on_vol_opacity_slider_changed(self, val: int):
        self.set_volume_opacity(val / 100.0)

    def set_smoothing_enabled(self, enabled: bool):
        """Enables or disables 3D surface mesh smoothing (affects rendering only)."""
        if self.smoothing_enabled != enabled:
            self.smoothing_enabled = enabled
            self.chk_smooth.blockSignals(True)
            self.chk_smooth.setChecked(enabled)
            self.chk_smooth.blockSignals(False)
            self._actor_voxel_counts.clear()  # Invalidate cached meshes to regenerate
            self._rebuild_all_segment_surfaces()

    # =========================================================================
    # Convenient Segment Actions
    # =========================================================================

    def show_all_segments(self):
        """Shows all available 3D segment surfaces together."""
        if self.segmentation_manager and self.segmentation_manager.model:
            for s_id in self.segmentation_manager.model.segments.keys():
                self._segment_3d_visibility[s_id] = True
        self.set_display_mode(self.MODE_ALL_SEGMENTS)
        self.refresh_3d_segments_table()

    def hide_all_segments(self):
        """Hides all 3D segment surfaces."""
        if self.segmentation_manager and self.segmentation_manager.model:
            for s_id in self.segmentation_manager.model.segments.keys():
                self._segment_3d_visibility[s_id] = False
        self._update_actor_visibilities()
        self.refresh_3d_segments_table()

    def show_selected_only(self, segment_id: Optional[int] = None):
        """Displays only the selected segment in 3D, hiding other surfaces."""
        if segment_id is not None and self.segmentation_manager:
            self.segmentation_manager.set_active_segment(segment_id)

        active_id = self.segmentation_manager.active_segment_id if self.segmentation_manager else None
        if active_id is not None:
            self._segment_3d_visibility[active_id] = True

        self.set_display_mode(self.MODE_SELECTED_ONLY)
        self.refresh_3d_segments_table()

    def set_segment_3d_visibility(self, segment_id: int, visible: bool):
        """Toggles individual segment 3D rendering without modifying mask data."""
        self._segment_3d_visibility[segment_id] = visible
        self._update_actor_visibilities()
        self.refresh_3d_segments_table()

    # =========================================================================
    # Transfer Functions and Presets
    # =========================================================================

    def _apply_volume_opacity_transfer(self):
        """Scales base preset transfer points by volume_opacity_multiplier."""
        if not self.opacity_tf:
            return
        self.opacity_tf.RemoveAllPoints()
        for scalar_val, base_op in self._base_opacity_points:
            self.opacity_tf.AddPoint(scalar_val, base_op * self.volume_opacity_multiplier)

    def _on_preset_changed(self, idx: int):
        preset = self.combo_preset.currentText()
        if preset == "CT Bone":
            self._apply_ct_bone_preset()
        elif preset == "CT Soft Tissue":
            self._apply_ct_soft_tissue_preset()
        elif preset == "CT Lung":
            self._apply_ct_lung_preset()
        elif preset == "MIP (Maximum Intensity)":
            self._apply_mip_preset()
        elif preset == "MR / Grayscale":
            self._apply_mr_preset()

        self.vtkWidget.GetRenderWindow().Render()

    def _apply_ct_bone_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        self._base_opacity_points = [
            (-1000, 0.0),
            (100, 0.0),
            (220, 0.12),
            (450, 0.55),
            (1200, 0.90)
        ]
        self._apply_volume_opacity_transfer()

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(120, 0.65, 0.45, 0.35)
        self.color_tf.AddRGBPoint(300, 0.88, 0.82, 0.70)
        self.color_tf.AddRGBPoint(800, 0.96, 0.94, 0.88)
        self.color_tf.AddRGBPoint(1500, 1.0, 1.0, 1.0)

    def _apply_ct_soft_tissue_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        self._base_opacity_points = [
            (-1000, 0.0),
            (-120, 0.0),
            (-40, 0.08),
            (40, 0.35),
            (150, 0.70),
            (800, 0.90)
        ]
        self._apply_volume_opacity_transfer()

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(-80, 0.25, 0.05, 0.05)   # Fat
        self.color_tf.AddRGBPoint(40, 0.80, 0.30, 0.25)    # Muscle/Parenchyma
        self.color_tf.AddRGBPoint(120, 0.95, 0.60, 0.45)   # Enhanced organ
        self.color_tf.AddRGBPoint(600, 0.98, 0.95, 0.90)   # Bone

    def _apply_ct_lung_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        self._base_opacity_points = [
            (-1000, 0.0),
            (-920, 0.0),
            (-750, 0.30),
            (-400, 0.10),
            (-100, 0.0),
            (400, 0.40)
        ]
        self._apply_volume_opacity_transfer()

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(-850, 0.20, 0.60, 0.90)  # Airways/Lung
        self.color_tf.AddRGBPoint(-600, 0.40, 0.85, 1.00)
        self.color_tf.AddRGBPoint(400, 0.95, 0.92, 0.85)   # Rib cage outline

    def _apply_mip_preset(self):
        self.volume_mapper.SetBlendModeToMaximumIntensity()
        self.volume_property.ShadeOff()

        self._base_opacity_points = [
            (-1000, 0.0),
            (50, 0.0),
            (300, 0.60),
            (1000, 1.00)
        ]
        self._apply_volume_opacity_transfer()

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(100, 0.4, 0.4, 0.4)
        self.color_tf.AddRGBPoint(500, 0.8, 0.8, 0.8)
        self.color_tf.AddRGBPoint(1200, 1.0, 1.0, 1.0)

    def _apply_mr_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_property.ShadeOn()

        val_min = 0.0
        val_max = 500.0
        if self.processor and self.processor.volume is not None:
            val_min = float(np.percentile(self.processor.volume, 5))
            val_max = float(np.percentile(self.processor.volume, 98))

        span = max(10.0, val_max - val_min)
        p1 = val_min + span * 0.15
        p2 = val_min + span * 0.45
        p3 = val_min + span * 0.85

        self._base_opacity_points = [
            (val_min, 0.0),
            (p1, 0.0),
            (p2, 0.35),
            (p3, 0.80)
        ]
        self._apply_volume_opacity_transfer()

        self.color_tf.RemoveAllPoints()
        self.color_tf.AddRGBPoint(val_min, 0.0, 0.0, 0.0)
        self.color_tf.AddRGBPoint(p1, 0.2, 0.2, 0.2)
        self.color_tf.AddRGBPoint(p2, 0.6, 0.6, 0.6)
        self.color_tf.AddRGBPoint(p3, 0.95, 0.95, 0.95)

    def _on_quality_changed(self, idx: int):
        txt = self.combo_quality.currentText()
        if "High" in txt:
            self.volume_mapper.SetSampleDistance(0.5)
        elif "Normal" in txt:
            self.volume_mapper.SetSampleDistance(1.0)
        else:
            self.volume_mapper.SetSampleDistance(2.0)
        self.vtkWidget.GetRenderWindow().Render()

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            self.setStyleSheet("border: 2px solid #00aaff;")
        else:
            self.setStyleSheet("border: 1px solid #2d2d2d;")

    def mousePressEvent(self, event):
        self.activated.emit("3D_VOLUME")
        super().mousePressEvent(event)

    # =========================================================================
    # Segmentation Synchronization & Surface Management
    # =========================================================================

    def set_segmentation_manager(self, manager):
        """Connects segmentation manager to 3D surface renderer."""
        self.segmentation_manager = manager
        if manager:
            manager.segment_updated.connect(self._on_segment_updated)
            manager.model_changed.connect(self._rebuild_all_segment_surfaces)
            manager.segment_list_changed.connect(self._on_segment_list_changed)
            manager.active_segment_changed.connect(self._on_active_segment_changed)

    def _on_active_segment_changed(self, segment_id: int):
        """Selects active segment and updates 3D view if in Selected Only mode."""
        self._highlight_table_row(segment_id)
        if self.display_mode == self.MODE_SELECTED_ONLY:
            self._update_actor_visibilities()

    def _on_segment_updated(self, segment_id: int):
        """Regenerates or updates 3D surface mesh for a specific segment."""
        if not self.processor or not self.processor.geometry or not self.segmentation_manager:
            return
        model = self.segmentation_manager.model
        if not model:
            return

        seg = model.get_segment(segment_id)
        if not seg:
            # Segment removed
            if segment_id in self._segment_actors:
                self.renderer.RemoveActor(self._segment_actors[segment_id])
                del self._segment_actors[segment_id]
                self._actor_voxel_counts.pop(segment_id, None)
                self._segment_3d_visibility.pop(segment_id, None)
                self._update_actor_visibilities()
            self.refresh_3d_segments_table()
            return

        # Sync 3D visibility with segment visibility
        self._segment_3d_visibility[segment_id] = seg.visible

        # If actor exists, update color and opacity
        if segment_id in self._segment_actors:
            actor = self._segment_actors[segment_id]
            r, g, b, _ = seg.color
            actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
            actor.GetProperty().SetOpacity(seg.opacity)

        # Check if surface mesh geometry needs re-extraction
        cached_count = self._actor_voxel_counts.get(segment_id)
        if cached_count != seg.voxel_count:
            # Remove previous actor if existed
            if segment_id in self._segment_actors:
                self.renderer.RemoveActor(self._segment_actors[segment_id])
                del self._segment_actors[segment_id]

            if seg.voxel_count > 0:
                new_actor = SegmentSurfaceExtractor.create_vtk_actor_from_segment(
                    seg, self.processor.geometry, smooth=self.smoothing_enabled
                )
                if new_actor is not None:
                    self.renderer.AddActor(new_actor)
                    self._segment_actors[segment_id] = new_actor
                    self._actor_voxel_counts[segment_id] = seg.voxel_count
            else:
                self._actor_voxel_counts.pop(segment_id, None)

        self._update_actor_visibilities()
        self.refresh_3d_segments_table()

    def _on_segment_list_changed(self):
        """Removes actors for deleted segments and refreshes 3D table."""
        if not self.segmentation_manager or not self.segmentation_manager.model:
            return
        current_ids = set(self.segmentation_manager.model.segments.keys())
        stale_ids = [s_id for s_id in list(self._segment_actors.keys()) if s_id not in current_ids]
        for s_id in stale_ids:
            self.renderer.RemoveActor(self._segment_actors[s_id])
            del self._segment_actors[s_id]
            self._actor_voxel_counts.pop(s_id, None)
            self._segment_3d_visibility.pop(s_id, None)

        self.refresh_3d_segments_table()
        self._update_actor_visibilities()

    def _rebuild_all_segment_surfaces(self):
        """Reconstructs all 3D segment surfaces (e.g. after project load, clear, or smoothing change)."""
        if not self.renderer:
            return
        # Remove old actors
        for actor in self._segment_actors.values():
            self.renderer.RemoveActor(actor)
        self._segment_actors.clear()
        self._actor_voxel_counts.clear()

        if not self.processor or not self.processor.geometry or not self.segmentation_manager:
            self.vtkWidget.GetRenderWindow().Render()
            return

        model = self.segmentation_manager.model
        if not model:
            self.vtkWidget.GetRenderWindow().Render()
            return

        for seg_id, seg in model.segments.items():
            if seg_id not in self._segment_3d_visibility:
                self._segment_3d_visibility[seg_id] = seg.visible

            if seg.voxel_count > 0:
                actor = SegmentSurfaceExtractor.create_vtk_actor_from_segment(
                    seg, self.processor.geometry, smooth=self.smoothing_enabled
                )
                if actor:
                    self.renderer.AddActor(actor)
                    self._segment_actors[seg_id] = actor
                    self._actor_voxel_counts[seg_id] = seg.voxel_count

        self._update_actor_visibilities()
        self.refresh_3d_segments_table()

    def _update_actor_visibilities(self):
        """Updates VTK actor visibilities based on display mode and individual segment toggles."""
        if not self.renderer:
            return

        # 1. Update Volume Actor Visibility
        vol_should_show = False
        if self.volume_actor:
            if self.display_mode in [self.MODE_VOLUME_ONLY, self.MODE_COMBINED]:
                vol_should_show = self.volume_visible
            elif self.display_mode == self.MODE_SELECTED_ONLY:
                vol_should_show = self.volume_visible
            else:  # MODE_ALL_SEGMENTS
                vol_should_show = False

            self.volume_actor.SetVisibility(1 if vol_should_show else 0)

        # 2. Update Segment Surface Actors Visibility
        active_id = self.segmentation_manager.active_segment_id if self.segmentation_manager else None
        model = self.segmentation_manager.model if self.segmentation_manager else None

        for seg_id, actor in self._segment_actors.items():
            seg = model.get_segment(seg_id) if model else None
            if not seg or seg.voxel_count == 0:
                actor.SetVisibility(0)
                continue

            is_3d_vis = self._segment_3d_visibility.get(seg_id, seg.visible)

            if self.display_mode == self.MODE_VOLUME_ONLY:
                actor.SetVisibility(0)
            elif self.display_mode == self.MODE_SELECTED_ONLY:
                # Mode 1: Show Selected Only
                if seg_id == active_id and is_3d_vis:
                    actor.SetVisibility(1)
                else:
                    actor.SetVisibility(0)
            elif self.display_mode == self.MODE_ALL_SEGMENTS:
                # Mode 2: Show All Segments
                actor.SetVisibility(1 if is_3d_vis else 0)
            elif self.display_mode == self.MODE_COMBINED:
                # Mode 3: Original Volume + Visible Segments
                actor.SetVisibility(1 if is_3d_vis else 0)

        if self.vtkWidget and self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()

    # =========================================================================
    # 3D Segment Visibility Table UI
    # =========================================================================

    def refresh_3d_segments_table(self):
        """Populates the 3D Display Panel segment list table with controls."""
        if not self.segmentation_manager or not self.segmentation_manager.model:
            self.table_3d_segments.setRowCount(0)
            return

        model = self.segmentation_manager.model
        segments = list(model.segments.values())
        self.table_3d_segments.blockSignals(True)
        self.table_3d_segments.setRowCount(len(segments))

        for row, seg in enumerate(segments):
            seg_id = seg.id
            is_3d_vis = self._segment_3d_visibility.get(seg_id, seg.visible)

            # Col 0: 3D Visibility Checkbox
            chk = QCheckBox()
            chk.setChecked(is_3d_vis)
            chk.setToolTip(f"Toggle 3D visibility of {seg.name}")
            chk.toggled.connect(lambda val, s_id=seg_id: self.set_segment_3d_visibility(s_id, val))
            chk_container = QWidget()
            chk_layout = QHBoxLayout(chk_container)
            chk_layout.setContentsMargins(4, 0, 4, 0)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk_layout.addWidget(chk)
            self.table_3d_segments.setCellWidget(row, 0, chk_container)

            # Col 1: Color Swatch Button
            r, g, b, a = seg.color
            btn_col = QToolButton()
            btn_col.setFixedSize(18, 18)
            btn_col.setStyleSheet(f"background-color: rgba({r},{g},{b},{a/255.0:.2f}); border: 1px solid #555566; border-radius: 2px;")
            btn_col.setToolTip(f"Change color of {seg.name}")
            btn_col.clicked.connect(lambda _, s_id=seg_id: self._pick_segment_color(s_id))
            col_container = QWidget()
            col_layout = QHBoxLayout(col_container)
            col_layout.setContentsMargins(4, 0, 4, 0)
            col_layout.setAlignment(Qt.AlignCenter)
            col_layout.addWidget(btn_col)
            self.table_3d_segments.setCellWidget(row, 1, col_container)

            # Col 2: Opacity Slider
            slider_op = QSlider(Qt.Horizontal)
            slider_op.setRange(5, 100)
            slider_op.setValue(int(round(seg.opacity * 100)))
            slider_op.setToolTip(f"Adjust 3D surface opacity of {seg.name}")
            slider_op.setStyleSheet("""
                QSlider::groove:horizontal { height: 3px; background: #333333; border-radius: 1px; }
                QSlider::sub-page:horizontal { background: #00d2ff; border-radius: 1px; }
                QSlider::handle:horizontal { background: #ffffff; width: 8px; margin-top: -2px; margin-bottom: -2px; border-radius: 4px; }
            """)
            slider_op.valueChanged.connect(lambda val, s_id=seg_id: self._on_segment_opacity_slider(s_id, val))
            op_container = QWidget()
            op_layout = QHBoxLayout(op_container)
            op_layout.setContentsMargins(2, 0, 2, 0)
            op_layout.addWidget(slider_op)
            self.table_3d_segments.setCellWidget(row, 2, op_container)

            # Col 3: Segment Name
            name_item = QTableWidgetItem(f"{seg.name} ({seg.calculate_volume_cm3(self.processor.geometry if self.processor else model.geometry):.1f} cm³)")
            name_item.setData(Qt.UserRole, seg_id)
            self.table_3d_segments.setItem(row, 3, name_item)

        self.table_3d_segments.blockSignals(False)
        self._highlight_table_row(self.segmentation_manager.active_segment_id)

    def _highlight_table_row(self, segment_id: Optional[int]):
        """Highlights the active segment row in the 3D table."""
        if segment_id is None:
            return
        self.table_3d_segments.blockSignals(True)
        for r in range(self.table_3d_segments.rowCount()):
            item = self.table_3d_segments.item(r, 3)
            if item and item.data(Qt.UserRole) == segment_id:
                self.table_3d_segments.selectRow(r)
                break
        self.table_3d_segments.blockSignals(False)

    def _on_table_row_selected(self):
        """Sets active segment in manager when clicked in 3D table."""
        row = self.table_3d_segments.currentRow()
        if row < 0:
            return
        item = self.table_3d_segments.item(row, 3)
        if item and self.segmentation_manager:
            seg_id = item.data(Qt.UserRole)
            if seg_id is not None and seg_id != self.segmentation_manager.active_segment_id:
                self.segmentation_manager.set_active_segment(seg_id)

    def _pick_segment_color(self, segment_id: int):
        """Opens color picker dialog for segment."""
        if not self.segmentation_manager or not self.segmentation_manager.model:
            return
        seg = self.segmentation_manager.model.get_segment(segment_id)
        if not seg:
            return
        r, g, b, a = seg.color
        init_color = QColor(r, g, b, a)
        chosen = QColorDialog.getColor(init_color, self, f"Select Color for {seg.name}")
        if chosen.isValid():
            new_rgba = (chosen.red(), chosen.green(), chosen.blue(), chosen.alpha())
            seg.color = new_rgba
            if segment_id in self._segment_actors:
                self._segment_actors[segment_id].GetProperty().SetColor(
                    new_rgba[0] / 255.0, new_rgba[1] / 255.0, new_rgba[2] / 255.0
                )
                self.vtkWidget.GetRenderWindow().Render()
            self.segmentation_manager.segment_updated.emit(segment_id)

    def _on_segment_opacity_slider(self, segment_id: int, val: int):
        """Updates segment surface opacity in 3D viewer."""
        if not self.segmentation_manager or not self.segmentation_manager.model:
            return
        seg = self.segmentation_manager.model.get_segment(segment_id)
        if not seg:
            return
        seg.opacity = val / 100.0
        if segment_id in self._segment_actors:
            self._segment_actors[segment_id].GetProperty().SetOpacity(seg.opacity)
            self.vtkWidget.GetRenderWindow().Render()
