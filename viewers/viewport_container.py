"""
2x2 Synchronized Medical Viewport Grid Container.
Coordinates Axial, Sagittal, Coronal, and 3D Volume viewports with linked crosshairs
and multi-planar physical coordinates.
"""
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter

from viewers.axial_viewer import AxialViewer
from viewers.sagittal_viewer import SagittalViewer
from viewers.coronal_viewer import CoronalViewer
from viewers.volume_3d_viewer import Volume3DViewer
from imaging.volume_processor import VolumeProcessor


class ViewportContainer(QWidget):
    """Holds the 2-by-2 grid of viewports with resizable splitters and linking logic."""
    active_viewport_changed = Signal(str)
    pixel_inspected = Signal(str, float, float, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor: Optional[VolumeProcessor] = None
        self.is_linked_navigation: bool = True
        self.active_plane: str = "AXIAL"

        # Voxel coordinates of current 3D crosshair position (z, y, x)
        self.curr_z: int = 0
        self.curr_y: int = 0
        self.curr_x: int = 0

        self.segmentation_manager = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main vertical splitter
        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.setStyleSheet("QSplitter::handle { background-color: #242424; }")

        # Top horizontal splitter (Axial, Sagittal)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.axial_view = AxialViewer()
        self.sagittal_view = SagittalViewer()
        self.top_splitter.addWidget(self.axial_view)
        self.top_splitter.addWidget(self.sagittal_view)

        # Bottom horizontal splitter (Coronal, 3D Volume)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.coronal_view = CoronalViewer()
        self.volume_3d_view = Volume3DViewer()
        self.bottom_splitter.addWidget(self.coronal_view)
        self.bottom_splitter.addWidget(self.volume_3d_view)

        self.v_splitter.addWidget(self.top_splitter)
        self.v_splitter.addWidget(self.bottom_splitter)

        # Set equal 50/50 proportions
        self.top_splitter.setSizes([500, 500])
        self.bottom_splitter.setSizes([500, 500])
        self.v_splitter.setSizes([500, 500])

        layout.addWidget(self.v_splitter)

    def _connect_signals(self):
        # Activation signals
        self.axial_view.activated.connect(self._set_active_plane)
        self.sagittal_view.activated.connect(self._set_active_plane)
        self.coronal_view.activated.connect(self._set_active_plane)
        self.volume_3d_view.activated.connect(self._set_active_plane)

        # Slice changed signals from sliders
        self.axial_view.slice_changed.connect(self._on_axial_slice_changed)
        self.sagittal_view.slice_changed.connect(self._on_sagittal_slice_changed)
        self.coronal_view.slice_changed.connect(self._on_coronal_slice_changed)

        # Crosshair click/drag events from viewports
        self.axial_view.viewport.crosshair_clicked.connect(self._on_axial_crosshair_clicked)
        self.sagittal_view.viewport.crosshair_clicked.connect(self._on_sagittal_crosshair_clicked)
        self.coronal_view.viewport.crosshair_clicked.connect(self._on_coronal_crosshair_clicked)

        # Window/Level synchronization
        self.axial_view.viewport.window_level_changed.connect(self._on_window_level_changed)
        self.sagittal_view.viewport.window_level_changed.connect(self._on_window_level_changed)
        self.coronal_view.viewport.window_level_changed.connect(self._on_window_level_changed)

        # Pixel inspection
        self.axial_view.viewport.pixel_inspected.connect(lambda c, r, v: self.pixel_inspected.emit("Axial", c, r, v))
        self.sagittal_view.viewport.pixel_inspected.connect(lambda c, r, v: self.pixel_inspected.emit("Sagittal", c, r, v))
        self.coronal_view.viewport.pixel_inspected.connect(lambda c, r, v: self.pixel_inspected.emit("Coronal", c, r, v))

    def set_active_plane(self, plane_name: str):
        self._set_active_plane(plane_name)

    def _set_active_plane(self, plane: str):
        self.active_plane = plane.upper()
        self.axial_view.set_active(self.active_plane == "AXIAL")
        self.sagittal_view.set_active(self.active_plane == "SAGITTAL")
        self.coronal_view.set_active(self.active_plane == "CORONAL")
        self.volume_3d_view.set_active(self.active_plane == "3D_VOLUME")
        self.active_viewport_changed.emit(self.active_plane)

    def set_segmentation_manager(self, manager):
        self.segmentation_manager = manager
        self.axial_view.set_segmentation_manager(manager)
        self.sagittal_view.set_segmentation_manager(manager)
        self.coronal_view.set_segmentation_manager(manager)
        self.volume_3d_view.set_segmentation_manager(manager)

    def set_processor(self, processor: VolumeProcessor):
        self.processor = processor
        if not processor.has_data():
            return

        if self.segmentation_manager and processor.geometry:
            self.segmentation_manager.set_geometry(processor.geometry)

        # Initialize default voxel center
        nz = processor.geometry.num_z
        ny = processor.geometry.num_y
        nx = processor.geometry.num_x

        self.curr_z = nz // 2
        self.curr_y = ny // 2
        self.curr_x = nx // 2

        # Populate all 4 viewports
        self.axial_view.set_processor(processor)
        self.sagittal_view.set_processor(processor)
        self.coronal_view.set_processor(processor)
        self.volume_3d_view.set_processor(processor)

        self._update_all_crosshairs()
        self._set_active_plane("AXIAL")

    def _on_axial_slice_changed(self, plane: str, slice_idx: int):
        self.curr_z = slice_idx
        self.axial_view.show_slice(slice_idx)
        if self.is_linked_navigation:
            self._update_all_crosshairs()

    def _on_sagittal_slice_changed(self, plane: str, slice_idx: int):
        self.curr_x = slice_idx
        self.sagittal_view.show_slice(slice_idx)
        if self.is_linked_navigation:
            self._update_all_crosshairs()

    def _on_coronal_slice_changed(self, plane: str, slice_idx: int):
        self.curr_y = slice_idx
        self.coronal_view.show_slice(slice_idx)
        if self.is_linked_navigation:
            self._update_all_crosshairs()

    def _on_axial_crosshair_clicked(self, col: float, row: float):
        # In axial slice: col is X, row is Y
        self.curr_x = int(round(col))
        self.curr_y = int(round(row))
        if self.is_linked_navigation and self.processor:
            self.sagittal_view.set_current_slice(self.curr_x)
            self.sagittal_view.show_slice(self.curr_x)
            self.coronal_view.set_current_slice(self.curr_y)
            self.coronal_view.show_slice(self.curr_y)
            self._update_all_crosshairs()

    def _on_sagittal_crosshair_clicked(self, col: float, row: float):
        # In sagittal slice: col is Y, row is inverted Z
        if not self.processor:
            return
        nz = self.processor.geometry.num_z
        self.curr_y = int(round(col))
        self.curr_z = int(round((nz - 1) - row))
        if self.is_linked_navigation:
            self.axial_view.set_current_slice(self.curr_z)
            self.axial_view.show_slice(self.curr_z)
            self.coronal_view.set_current_slice(self.curr_y)
            self.coronal_view.show_slice(self.curr_y)
            self._update_all_crosshairs()

    def _on_coronal_crosshair_clicked(self, col: float, row: float):
        # In coronal slice: col is X, row is inverted Z
        if not self.processor:
            return
        nz = self.processor.geometry.num_z
        self.curr_x = int(round(col))
        self.curr_z = int(round((nz - 1) - row))
        if self.is_linked_navigation:
            self.axial_view.set_current_slice(self.curr_z)
            self.axial_view.show_slice(self.curr_z)
            self.sagittal_view.set_current_slice(self.curr_x)
            self.sagittal_view.show_slice(self.curr_x)
            self._update_all_crosshairs()

    def _update_all_crosshairs(self):
        if not self.processor or not self.processor.has_data():
            return
        self.axial_view.update_crosshair(self.curr_x, self.curr_y)
        self.sagittal_view.update_crosshair(self.curr_y, self.curr_z)
        self.coronal_view.update_crosshair(self.curr_x, self.curr_z)

    def _on_window_level_changed(self, window: float, level: float):
        """Synchronizes window/level across all three 2D orthogonal planes."""
        self.axial_view.viewport.set_window_level(window, level)
        self.sagittal_view.viewport.set_window_level(window, level)
        self.coronal_view.viewport.set_window_level(window, level)

    def apply_window_preset(self, preset_name: str):
        if self.processor:
            self.processor.window_level_manager.apply_preset(preset_name)
            w = self.processor.window_level_manager.window
            l = self.processor.window_level_manager.level
            self._on_window_level_changed(w, l)

    def set_active_tool(self, tool_name: str):
        """Sets interaction tool ('window_level', 'zoom', 'pan', 'measure', 'crosshair', 'segment_brush', 'segment_eraser', 'segment_contour', 'segment_seed_picker')."""
        self.axial_view.viewport.active_tool = tool_name
        self.sagittal_view.viewport.active_tool = tool_name
        self.coronal_view.viewport.active_tool = tool_name
        if self.segmentation_manager:
            if tool_name == "segment_brush":
                self.segmentation_manager.set_active_tool("brush")
            elif tool_name == "segment_eraser":
                self.segmentation_manager.set_active_tool("eraser")
            elif tool_name == "segment_contour":
                self.segmentation_manager.set_active_tool("contour")
            elif tool_name == "segment_seed_picker":
                self.segmentation_manager.set_active_tool("seed_picker")
            else:
                self.segmentation_manager.set_active_tool("none")
        self.axial_view.viewport.update()
        self.sagittal_view.viewport.update()
        self.coronal_view.viewport.update()

    def toggle_crosshair_link(self, enabled: bool):
        self.is_linked_navigation = enabled
        self.axial_view.viewport.show_crosshair = enabled
        self.sagittal_view.viewport.show_crosshair = enabled
        self.coronal_view.viewport.show_crosshair = enabled
        self.axial_view.viewport.update()
        self.sagittal_view.viewport.update()
        self.coronal_view.viewport.update()

    def reset_layout(self):
        """Resets splitters to equal 50/50 proportion."""
        w = self.width() // 2
        h = self.height() // 2
        self.top_splitter.setSizes([w, w])
        self.bottom_splitter.setSizes([w, w])
        self.v_splitter.setSizes([h, h])

    def reset_views(self):
        self.axial_view.viewport.reset_view()
        self.sagittal_view.viewport.reset_view()
        self.coronal_view.viewport.reset_view()
        self.volume_3d_view.reset_camera()
