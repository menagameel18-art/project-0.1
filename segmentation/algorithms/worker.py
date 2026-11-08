"""
Asynchronous Segmentation Background Worker.
Executes computationally intensive semi-automatic operations (thresholding, region growing)
without blocking the Qt user interface thread.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from PySide6.QtCore import QThread, Signal

from imaging.coordinate_manager import VolumeGeometry
from segmentation.provider import SegmentationProvider


class SemiAutoWorker(QThread):
    """Background worker thread for running segmentation algorithms asynchronously."""
    progress = Signal(int, str)       # percent (0-100), status text
    result_ready = Signal(object)     # np.ndarray mask
    failed = Signal(str)              # error string
    cancelled = Signal()

    def __init__(
        self,
        provider: SegmentationProvider,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        parameters: Dict[str, Any],
        parent=None
    ):
        super().__init__(parent)
        self.provider = provider
        self.volume = volume
        self.geometry = geometry
        self.parameters = parameters
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        try:
            self._is_cancelled = False

            def progress_cb(pct: int, msg: str):
                if not self._is_cancelled:
                    self.progress.emit(pct, msg)

            def cancel_cb() -> bool:
                return self._is_cancelled

            result_mask = self.provider.execute(
                volume=self.volume,
                geometry=self.geometry,
                parameters=self.parameters,
                progress_callback=progress_cb,
                cancel_check=cancel_cb
            )

            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.result_ready.emit(result_mask)

        except Exception as e:
            if not self._is_cancelled:
                self.failed.emit(str(e))
