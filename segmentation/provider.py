"""
Extensible Segmentation Provider Interface.
Defines clean abstract interfaces for segmentation methods (Manual, Semi-Automatic,
and future AI segmentation providers like MONAI, TotalSegmentator, or nnUNet).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import numpy as np

from imaging.coordinate_manager import VolumeGeometry


class SegmentationProvider(ABC):
    """Abstract base class for all segmentation providers."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique machine identifier, e.g. 'manual_brush', 'semi_auto_threshold', 'ai_totalseg'."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'Manual Brush', 'Region Growing', 'AI Organ Seg'."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category: 'manual', 'semi_automatic', or 'ai'."""
        pass

    @property
    def description(self) -> str:
        return ""

    @property
    def is_interactive(self) -> bool:
        """True if provider responds to continuous user mouse interaction."""
        return False

    @property
    def is_asynchronous(self) -> bool:
        """True if execution should run in a background worker thread."""
        return False

    @abstractmethod
    def execute(
        self,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> np.ndarray:
        """
        Executes segmentation algorithm and returns 3D binary mask (Z, Y, X).
        """
        pass
