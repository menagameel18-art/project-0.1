"""
Sample Data Manager and Asynchronous Downloader.
Manages curated public medical case library, manifests, and procedural demo phantoms.
"""
from __future__ import annotations
import os
import json
import zipfile
import tarfile
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
import requests
from PySide6.QtCore import QObject, Signal, QThread

from imaging.volume_processor import VolumeProcessor


@dataclass
class SampleCase:
    id: str
    title: str
    modality: str
    anatomical_region: str
    description: str
    size_str: str
    download_url: Optional[str] = None
    is_procedural_demo: bool = False
    is_downloaded: bool = False
    local_dir: str = ""
    license_info: str = "Public Medical Research / Open License"


# Built-in curated catalog of medical sample cases
BUILTIN_CASES: List[SampleCase] = [
    SampleCase(
        id="demo_head_ct",
        title="CT Head & Cranium (High Contrast)",
        modality="CT",
        anatomical_region="Head / Brain",
        description="Comprehensive 3D CT study of the neurocranium, brain parenchyma, ventricles, and facial bones. Instant offline procedural reconstruction.",
        size_str="~4 MB (Instant Procedural)",
        is_procedural_demo=True,
        is_downloaded=True,
        license_info="Antigravity Medical Open Phantom License"
    ),
    SampleCase(
        id="demo_chest_ct",
        title="CT Thorax & Chest Cavity",
        modality="CT",
        anatomical_region="Chest / Thorax",
        description="High-resolution axial thoracic CT series showing bilateral lungs, rib cage, thoracic spine, and heart/mediastinum. Instant offline procedural reconstruction.",
        size_str="~4 MB (Instant Procedural)",
        is_procedural_demo=True,
        is_downloaded=True,
        license_info="Antigravity Medical Open Phantom License"
    ),
    SampleCase(
        id="slicer_chest_ct",
        title="3D Slicer Chest CT Sample",
        modality="CT",
        anatomical_region="Chest / Thorax",
        description="Public benchmark clinical chest CT DICOM series from the official 3D Slicer medical repository.",
        size_str="~25 MB",
        download_url="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1bf7f59453c5e88ec7b09335a90d9f05a9643d9b071987ee965f5a0134bc5ec4",
        is_procedural_demo=False,
        is_downloaded=False,
        license_info="3D Slicer BSD-style Open License"
    ),
    SampleCase(
        id="mri_brain_t1",
        title="MRI Brain Neuroaxis (T1-Weighted)",
        modality="MR",
        anatomical_region="Brain",
        description="High-contrast T1-weighted neurological magnetic resonance study with clear grey/white matter differentiation.",
        size_str="~15 MB",
        download_url="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/e9391054b1f6920f781df5a0226462725e2aa6b0266e7ea81896068222fa141d",
        is_procedural_demo=False,
        is_downloaded=False,
        license_info="3D Slicer BSD-style Open License"
    ),
    SampleCase(
        id="ct_bone_ortho",
        title="CT Skeletal & Bone Extremity",
        modality="CT",
        anatomical_region="Skeletal / Bone",
        description="Diagnostic bone CT imaging optimized for orthopedic evaluation and 3D surface reconstruction.",
        size_str="~18 MB",
        download_url="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/2f77e68ad11c97a5448332d73359c25f463d1620fa6302caee43e62fcfec3907",
        is_procedural_demo=False,
        is_downloaded=False,
        license_info="Public Domain Medical Data"
    ),
    SampleCase(
        id="ct_abdomen_contrast",
        title="CT Abdomen & Pelvis",
        modality="CT",
        anatomical_region="Abdomen",
        description="Contrast-enhanced abdominal CT study displaying liver, kidneys, spleen, and abdominal aorta.",
        size_str="~32 MB",
        download_url="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/331a982f63f57ebbf7019f3595f9c464ef2448384b6f1228513bfbece0be1559",
        is_procedural_demo=False,
        is_downloaded=False,
        license_info="Public Research Dataset"
    ),
]


class SampleDownloadWorker(QThread):
    """Downloads medical sample archive asynchronously with progress updates."""
    progress = Signal(int, int, str)  # bytes_downloaded, total_bytes, message
    finished = Signal(str, str)       # case_id, local_extracted_dir
    failed = Signal(str, str)         # case_id, error_message

    def __init__(self, case: SampleCase, dest_dir: str):
        super().__init__()
        self.case = case
        self.dest_dir = dest_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if not self.case.download_url:
                raise ValueError("Case does not have a download URL.")

            os.makedirs(self.dest_dir, exist_ok=True)
            case_folder = os.path.join(self.dest_dir, self.case.id)
            os.makedirs(case_folder, exist_ok=True)

            archive_path = os.path.join(case_folder, "download.archive")

            resp = requests.get(self.case.download_url, stream=True, timeout=30)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(archive_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if self._is_cancelled:
                        self.failed.emit(self.case.id, "Download cancelled by user.")
                        return
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(
                            downloaded,
                            total_size,
                            f"Downloading {self.case.title}: {downloaded // 1024} KB / {total_size // 1024} KB"
                        )

            # Unpack if zip or tar
            if zipfile.is_zipfile(archive_path):
                self.progress.emit(downloaded, total_size, "Extracting DICOM files...")
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(case_folder)
            elif tarfile.is_tarfile(archive_path):
                self.progress.emit(downloaded, total_size, "Extracting DICOM files...")
                with tarfile.open(archive_path, "r") as tar_ref:
                    tar_ref.extractall(case_folder)

            self.finished.emit(self.case.id, case_folder)
        except Exception as e:
            self.failed.emit(self.case.id, str(e))


class SampleDataManager(QObject):
    """Manages sample cases catalog and local persistence."""
    case_download_started = Signal(str)
    case_download_progress = Signal(int, int, str)
    case_download_finished = Signal(str, str)
    case_download_failed = Signal(str, str)

    def __init__(self, base_dir: str = "sample_data"):
        super().__init__()
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.manifest_file = os.path.join(self.base_dir, "manifest.json")
        self.cases: List[SampleCase] = []
        self._active_worker: Optional[SampleDownloadWorker] = None
        self._init_catalog()

    def _init_catalog(self):
        # Load existing manifest if present
        saved_status = {}
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        saved_status[item["id"]] = item
            except Exception:
                saved_status = {}

        self.cases = []
        for case in BUILTIN_CASES:
            c = SampleCase(**asdict(case))
            local_case_dir = os.path.join(self.base_dir, c.id)
            if c.id in saved_status:
                c.is_downloaded = saved_status[c.id].get("is_downloaded", False)
                c.local_dir = saved_status[c.id].get("local_dir", local_case_dir)
            elif c.is_procedural_demo:
                c.is_downloaded = True
                c.local_dir = local_case_dir
            elif os.path.exists(local_case_dir) and any(os.scandir(local_case_dir)):
                c.is_downloaded = True
                c.local_dir = local_case_dir

            self.cases.append(c)

    def _save_manifest(self):
        try:
            items = [asdict(c) for c in self.cases]
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
        except Exception:
            pass

    def get_cases(self) -> List[SampleCase]:
        return self.cases

    def get_case_by_id(self, case_id: str) -> Optional[SampleCase]:
        for c in self.cases:
            if c.id == case_id:
                return c
        return None

    def load_case_processor(self, case_id: str) -> VolumeProcessor:
        """Loads or creates volume processor for a given case."""
        case = self.get_case_by_id(case_id)
        if not case:
            raise ValueError(f"Unknown case ID: {case_id}")

        if case.id == "demo_head_ct":
            return VolumeProcessor.create_procedural_head_phantom()
        elif case.id == "demo_chest_ct":
            return VolumeProcessor.create_procedural_chest_phantom()
        else:
            # Check local files
            from dicom.dicom_loader import DicomLoader
            if not case.local_dir or not os.path.exists(case.local_dir):
                raise ValueError(f"Case {case.title} is not downloaded yet.")

            series_map = DicomLoader.scan_directory(case.local_dir)
            if not series_map:
                raise ValueError(f"No DICOM series found in {case.local_dir}")

            first_series = next(iter(series_map.values()))
            vol, geom, meta = DicomLoader.load_series_volume(first_series)
            proc = VolumeProcessor()
            proc.set_volume_data(vol, geom, meta)
            return proc

    def download_case(self, case_id: str):
        case = self.get_case_by_id(case_id)
        if not case or not case.download_url:
            return

        self.case_download_started.emit(case.title)
        self._active_worker = SampleDownloadWorker(case, self.base_dir)
        self._active_worker.progress.connect(self.case_download_progress.emit)
        self._active_worker.finished.connect(self._on_download_finished)
        self._active_worker.failed.connect(self.case_download_failed.emit)
        self._active_worker.start()

    def cancel_download(self):
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.cancel()

    def _on_download_finished(self, case_id: str, local_dir: str):
        case = self.get_case_by_id(case_id)
        if case:
            case.is_downloaded = True
            case.local_dir = local_dir
            self._save_manifest()
            self.case_download_finished.emit(case_id, local_dir)
