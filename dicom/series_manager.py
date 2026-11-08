"""
Series manager and asynchronous background loading workers.
Maintains state of loaded studies, current series, and notifies UI components.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread

from dicom.dicom_loader import DicomLoader, DicomSeriesGroup
from imaging.coordinate_manager import VolumeGeometry
from imaging.volume_processor import SeriesMetadata, VolumeProcessor


class DirectoryScanWorker(QThread):
    """Background worker for recursive DICOM directory scanning."""
    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(dict)           # Dict[str, DicomSeriesGroup]
    failed = Signal(str)

    def __init__(self, directory_path: str):
        super().__init__()
        self.directory_path = directory_path

    def run(self):
        try:
            def on_progress(curr, tot, fname):
                self.progress.emit(curr, tot, fname)

            result = DicomLoader.scan_directory(self.directory_path, progress_callback=on_progress)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class SeriesLoadWorker(QThread):
    """Background worker for loading and assembling 3D DICOM series."""
    progress = Signal(int, int, str)
    finished = Signal(object, object, object)  # volume, geometry, metadata
    failed = Signal(str)

    def __init__(self, series_group: DicomSeriesGroup):
        super().__init__()
        self.series_group = series_group

    def run(self):
        try:
            def on_progress(curr, tot, msg):
                self.progress.emit(curr, tot, msg)

            vol, geom, meta = DicomLoader.load_series_volume(self.series_group, progress_callback=on_progress)
            self.finished.emit(vol, geom, meta)
        except Exception as e:
            self.failed.emit(str(e))


class SeriesManager(QObject):
    """Coordinates series selection, asynchronous loading, and volume updates."""
    series_map_updated = Signal(dict)
    loading_started = Signal(str)
    loading_progress = Signal(int, int, str)
    volume_loaded = Signal(object)  # VolumeProcessor
    loading_failed = Signal(str)

    def __init__(self, volume_processor: VolumeProcessor):
        super().__init__()
        self.volume_processor = volume_processor
        self.series_map: Dict[str, DicomSeriesGroup] = {}
        self.current_series_uid: Optional[str] = None
        self._scan_worker: Optional[DirectoryScanWorker] = None
        self._load_worker: Optional[SeriesLoadWorker] = None

    def scan_directory(self, folder_path: str):
        self.loading_started.emit(f"Scanning folder: {folder_path}...")
        self._scan_worker = DirectoryScanWorker(folder_path)
        self._scan_worker.progress.connect(self.loading_progress.emit)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _on_scan_finished(self, series_map: Dict[str, DicomSeriesGroup]):
        self.series_map = series_map
        self.series_map_updated.emit(series_map)
        if not series_map:
            self.loading_failed.emit("No valid DICOM files found in selected directory.")

    def _on_scan_failed(self, error_msg: str):
        self.loading_failed.emit(f"Error scanning directory: {error_msg}")

    def load_series(self, series_uid: str):
        if series_uid not in self.series_map:
            self.loading_failed.emit(f"Series UID {series_uid} not found.")
            return

        series = self.series_map[series_uid]
        self.current_series_uid = series_uid
        self.loading_started.emit(f"Loading series: {series.series_description} ({series.slice_count} slices)...")

        self._load_worker = SeriesLoadWorker(series)
        self._load_worker.progress.connect(self.loading_progress.emit)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.start()

    def _on_load_finished(self, volume: np.ndarray, geometry: VolumeGeometry, metadata: SeriesMetadata):
        self.volume_processor.set_volume_data(volume, geometry, metadata)
        self.volume_loaded.emit(self.volume_processor)

    def _on_load_failed(self, error_msg: str):
        self.loading_failed.emit(f"Failed to load series volume: {error_msg}")

    def load_direct_processor(self, processor: VolumeProcessor):
        """Loads an already constructed volume processor (e.g. sample case or demo phantom)."""
        self.volume_processor = processor
        self.volume_loaded.emit(self.volume_processor)
