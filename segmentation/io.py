"""
Segmentation I/O and Persistence.
Saves and loads validated multi-segment project files (.seg.npz) preserving
segment names, customizable colors, opacities, and 3D physical coordinate mapping.
"""
from __future__ import annotations
import json
import io
import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np

from imaging.coordinate_manager import VolumeGeometry
from segmentation.model import Segment, SegmentationModel


class SegmentationIO:
    """Manages serialization, deserialization, and geometry validation of segmentations."""

    FORMAT_VERSION = "1.0"

    @classmethod
    def save_to_file(
        cls,
        model: SegmentationModel,
        file_path: str,
        patient_id: str = "Unknown",
        series_uid: str = ""
    ) -> bool:
        """
        Saves all segments, colors, names, and volume geometry to a validated .seg.npz file.
        """
        g = model.geometry
        meta = {
            "format_version": cls.FORMAT_VERSION,
            "created_at": datetime.datetime.now().isoformat(),
            "patient_id": patient_id,
            "series_uid": series_uid,
            "geometry": {
                "shape": list(g.shape),
                "spacing": list(g.spacing),
                "origin": list(g.origin),
                "row_cosines": list(g.row_cosines),
                "col_cosines": list(g.col_cosines),
                "slice_normal": list(g.slice_normal),
            },
            "segments": []
        }

        arrays_to_save: Dict[str, np.ndarray] = {}

        for seg_id, seg in sorted(model.segments.items()):
            meta["segments"].append({
                "id": seg.id,
                "name": seg.name,
                "color": list(seg.color),
                "opacity": float(seg.opacity),
                "visible": bool(seg.visible),
                "voxel_count": seg.voxel_count
            })
            # Store compressed binary mask for each segment
            arrays_to_save[f"mask_{seg.id}"] = np.packbits(seg.mask.astype(bool))

        # Also store consolidated integer labelmap for external tools
        arrays_to_save["labelmap"] = model.get_labelmap()

        # Save metadata as JSON inside npz archive
        meta_json_bytes = json.dumps(meta, indent=2).encode("utf-8")
        arrays_to_save["metadata_json"] = np.frombuffer(meta_json_bytes, dtype=np.uint8)

        np.savez_compressed(file_path, **arrays_to_save)
        return True

    @classmethod
    def load_from_file(
        cls,
        file_path: str,
        current_geometry: Optional[VolumeGeometry] = None
    ) -> Tuple[SegmentationModel, Dict[str, Any]]:
        """
        Loads segmentation model from .seg.npz file.
        Validates volume geometry against current_geometry if provided.
        Returns:
            (model, metadata_dict)
        """
        npz = np.load(file_path, allow_pickle=False)

        if "metadata_json" not in npz:
            raise ValueError("Invalid segmentation file: missing metadata header.")

        meta_bytes = bytes(npz["metadata_json"])
        meta = json.loads(meta_bytes.decode("utf-8"))

        geom_info = meta.get("geometry", {})
        saved_shape = tuple(geom_info.get("shape", (1, 1, 1)))
        saved_spacing = tuple(geom_info.get("spacing", (1.0, 1.0, 1.0)))
        saved_origin = tuple(geom_info.get("origin", (0.0, 0.0, 0.0)))
        saved_row_cosines = tuple(geom_info.get("row_cosines", (1.0, 0.0, 0.0)))
        saved_col_cosines = tuple(geom_info.get("col_cosines", (0.0, 1.0, 0.0)))
        saved_slice_normal = tuple(geom_info.get("slice_normal", (0.0, 0.0, 1.0)))

        saved_geom = VolumeGeometry(
            shape=saved_shape,
            spacing=saved_spacing,
            origin=saved_origin,
            row_cosines=saved_row_cosines,
            col_cosines=saved_col_cosines,
            slice_normal=saved_slice_normal
        )

        target_geom = current_geometry or saved_geom

        # Strict validation of volume shape
        if current_geometry is not None and current_geometry.shape != saved_shape:
            raise ValueError(
                f"Geometry mismatch: segmentation volume shape {saved_shape} does not match active study shape {current_geometry.shape}."
            )

        model = SegmentationModel(target_geom)

        for s_data in meta.get("segments", []):
            s_id = int(s_data["id"])
            s_name = str(s_data.get("name", f"Segment {s_id}"))
            s_color = tuple(s_data.get("color", (255, 0, 0, 200)))
            s_opacity = float(s_data.get("opacity", 0.65))
            s_visible = bool(s_data.get("visible", True))

            mask_key = f"mask_{s_id}"
            if mask_key in npz:
                packed = npz[mask_key]
                total_voxels = int(np.prod(saved_shape))
                unpacked = np.unpackbits(packed)[:total_voxels].reshape(saved_shape).astype(np.uint8)
            else:
                unpacked = np.zeros(saved_shape, dtype=np.uint8)

            seg = Segment(
                id=s_id,
                name=s_name,
                color=s_color,
                visible=s_visible,
                opacity=s_opacity,
                mask=unpacked
            )
            model.segments[s_id] = seg
            model._next_id = max(model._next_id, s_id + 1)

        return model, meta
