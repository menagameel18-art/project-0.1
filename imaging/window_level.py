"""
Window / Level (contrast and brightness) mapping and standard clinical presets.
Supports Hounsfield Units (CT) and arbitrary intensity scales (MRI/PET/NM/synthetic).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class WindowPreset:
    name: str
    window: float
    level: float
    modality: str = "CT"


# Standard clinical CT presets according to radiology guidelines
STANDARD_CT_PRESETS: Dict[str, WindowPreset] = {
    "Soft Tissue / Abdomen": WindowPreset("Soft Tissue / Abdomen", window=400.0, level=40.0, modality="CT"),
    "Bone": WindowPreset("Bone", window=1800.0, level=400.0, modality="CT"),
    "Lung": WindowPreset("Lung", window=1500.0, level=-600.0, modality="CT"),
    "Brain": WindowPreset("Brain", window=80.0, level=40.0, modality="CT"),
    "Mediastinum": WindowPreset("Mediastinum", window=350.0, level=50.0, modality="CT"),
    "Subdural / Blood": WindowPreset("Subdural / Blood", window=130.0, level=50.0, modality="CT"),
    "Liver": WindowPreset("Liver", window=150.0, level=30.0, modality="CT"),
}


class WindowLevelManager:
    """Applies window/level transformation to convert image data to 8-bit display format."""

    def __init__(self, window: float = 400.0, level: float = 40.0):
        self._window: float = max(1.0, float(window))
        self._level: float = float(level)

    @property
    def window(self) -> float:
        return self._window

    @window.setter
    def window(self, val: float):
        self._window = max(1.0, float(val))

    @property
    def level(self) -> float:
        return self._level

    @level.setter
    def level(self, val: float):
        self._level = float(val)

    def set_window_level(self, window: float, level: float):
        self._window = max(1.0, float(window))
        self._level = float(level)

    def apply_preset(self, preset_name: str) -> bool:
        if preset_name in STANDARD_CT_PRESETS:
            p = STANDARD_CT_PRESETS[preset_name]
            self.set_window_level(p.window, p.level)
            return True
        return False

    def auto_window(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculates optimal window and level based on image histogram."""
        if data.size == 0:
            return (self._window, self._level)
        
        # Filter finite values
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            return (self._window, self._level)

        p1 = float(np.percentile(valid, 1))
        p99 = float(np.percentile(valid, 99))
        
        window = max(1.0, p99 - p1)
        level = (p99 + p1) / 2.0
        self.set_window_level(window, level)
        return (self._window, self._level)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Maps numerical array to uint8 [0, 255] using linear ramp windowing:
        Min = Level - Window / 2
        Max = Level + Window / 2
        """
        if data.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        w = self._window
        l = self._level
        
        low = l - (w / 2.0)
        high = l + (w / 2.0)

        # Vectorized linear mapping with clipping
        mapped = np.clip((data - low) / (high - low), 0.0, 1.0)
        return (mapped * 255.0).astype(np.uint8)
