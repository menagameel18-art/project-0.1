"""
Three-Dimensional Segmentation Surface Generation using VTK.
Extracts smooth isosurfaces from binary segment masks and preserves physical
volume coordinates, spacing, and anatomical origin.
"""
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

from imaging.coordinate_manager import VolumeGeometry
from segmentation.model import Segment


class SegmentSurfaceExtractor:
    """Generates and manages 3D polygonal surface meshes for segments."""

    @staticmethod
    def create_vtk_actor_from_segment(
        segment: Segment,
        geometry: VolumeGeometry,
        smooth: bool = True
    ) -> Optional[vtk.vtkActor]:
        """
        Extracts a smoothed 3D surface mesh from a segment binary mask.
        Returns a configured vtkActor or None if the segment is empty.
        Smoothing is optional and affects rendering only, never altering the saved mask array.
        """
        if segment.voxel_count == 0:
            return None

        nz, ny, nx = geometry.shape
        # Create VTK Image Data with physical origin and spacing
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(nx, ny, nz)
        vtk_image.SetSpacing(geometry.dx, geometry.dy, geometry.dz)
        vtk_image.SetOrigin(geometry.origin[0], geometry.origin[1], geometry.origin[2])

        # NumPy (Z, Y, X) flat array matches VTK (X fastest, then Y, then Z)
        flat_mask = np.ascontiguousarray(segment.mask, dtype=np.uint8).ravel()
        vtk_array = numpy_to_vtk(num_array=flat_mask, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_array.SetName("SegmentMask")
        vtk_image.GetPointData().SetScalars(vtk_array)

        # Fast isosurface extraction
        extractor = vtk.vtkFlyingEdges3D()
        extractor.SetInputData(vtk_image)
        extractor.SetValue(0, 0.5)
        extractor.ComputeNormalsOff()
        extractor.ComputeGradientsOff()
        extractor.Update()

        poly_data = extractor.GetOutput()
        if poly_data.GetNumberOfPoints() == 0:
            return None

        # Smooth surface mesh (optional, rendering only)
        if smooth:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(extractor.GetOutputPort())
            smoother.SetNumberOfIterations(15)
            smoother.BoundarySmoothingOn()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetPassBand(0.1)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            surface_source_port = smoother.GetOutputPort()
        else:
            surface_source_port = extractor.GetOutputPort()

        # Generate smooth surface normals for realistic shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(surface_source_port)
        normals.SetFeatureAngle(60.0)
        normals.ConsistencyOn()
        normals.SplittingOff()

        # Polygonal mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()

        # Surface Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Apply segment color and opacity
        r, g, b, _ = segment.color
        prop = actor.GetProperty()
        prop.SetColor(r / 255.0, g / 255.0, b / 255.0)
        prop.SetOpacity(segment.opacity)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(20.0)
        prop.SetAmbient(0.20)
        prop.SetDiffuse(0.80)
        prop.SetInterpolationToPhong()

        actor.SetVisibility(1 if segment.visible else 0)
        return actor
