# Antigravity Medical DICOM 3D Workstation

A professional, fully functional medical imaging workstation developed in Python 3.11 with **PySide6**, **VTK**, **SimpleITK**, **pydicom**, and **NumPy**. It loads, processes, and visualizes real DICOM CT and MRI series in synchronized 2D multiplanar orthogonal views (**Axial**, **Sagittal**, **Coronal**) and GPU/CPU ray-cast **3D Volume Rendering**.

---

## 📸 Workstation Layout & Features

- **2x2 Synchronized Viewport Grid**:
  - **Top-Left (Axial)**: Transverse physical cross-sections with mouse-wheel navigation and orientation indicators (**A**, **P**, **R**, **L**).
  - **Top-Right (Sagittal)**: Multiplanar reconstruction along Left-Right axis with anisotropic voxel aspect ratio compensation and orientation indicators (**S**, **I**, **A**, **P**).
  - **Bottom-Left (Coronal)**: Multiplanar reconstruction along Anterior-Posterior axis with orientation indicators (**S**, **I**, **R**, **L**).
  - **Bottom-Right (3D Volume)**: Hardware-accelerated VTK ray-cast 3D volume reconstruction with dynamic clinical color/opacity transfer functions and 3D orientation marker.
- **Independent Slice Navigation Sliders**:
  - Each 2D panel features an independent slider with physical slice position readout in millimeters (`Z: +15.2 mm`).
  - Responds directly to mouse-wheel scrolling over either the slider or the viewport canvas.
- **Physical Crosshair & Linked Navigation**:
  - Clicking or dragging crosshairs in any 2D plane calculates the exact physical coordinate $(X, Y, Z)$ in patient space (DICOM LPS) and immediately updates the intersecting orthogonal slices in the other viewports.
- **Radiological Clinical Tools**:
  - **Window / Level Adjustments**: Interactive left-drag contrast/brightness control and presets for CT Soft Tissue/Abdomen, Bone, Lung, Brain, Mediastinum, Liver, and Blood.
  - **Distance Caliper Measurement**: Click 2 points to measure physical lengths in millimeters calculated from voxel spacing.
  - **Pixel Intensity & Hounsfield Inspector**: Real-time HUD showing cursor coordinate and calibrated HU values (`+42.0 HU`).
  - **Pan, Zoom, and Fit**: Smooth pan, anchor-centered zoom, flip horizontal/vertical, and reset view.
- **Built-in Medical Case Library & Procedural Demonstrations**:
  - First-run welcome experience with 1-click launch of high-contrast 3D CT demonstration cases.
  - Asynchronous dataset download manager for benchmark public clinical datasets with progress tracking, cancellation, and validation.

---

## 🛠️ Technology Stack & Compatible Versions

| Component | Library / Version | Purpose |
| :--- | :--- | :--- |
| **GUI Framework** | PySide6 (`>= 6.6.0`) | Modern dark-themed radiology workstation UI, splitters, tree views |
| **DICOM Parsing** | pydicom (`>= 2.4.0`) | DICOM file header reading, transfer syntax validation, pixel arrays |
| **Volume Handling** | SimpleITK (`>= 2.3.0`) | Medical series handling, image resampling, geometry checks |
| **3D Rendering** | VTK (`>= 9.2.0`) | `vtkSmartVolumeMapper`, transfer functions, and 3D orientation markers |
| **Numerical Processing**| NumPy (`>= 1.24.0`) | Array manipulation, vectorized windowing, and coordinate algebra |
| **Networking** | Requests (`>= 2.31.0`) | Background download manager for sample dataset archives |
| **Testing** | PyTest (`>= 8.0.0`) | Unit and integration test suite |

---

## 📁 Project Architecture

```
c:/Users/User/Downloads/ide/
├── main.py                     # Workstation entry point and application lifecycle
├── requirements.txt           # Dependency specifications
├── pytest.ini                 # Test environment configuration
├── README.md                  # Comprehensive documentation and guide
├── dicom/
│   ├── dicom_loader.py        # Safe recursive DICOM scanning, slice ordering & HU assembly
│   ├── series_manager.py      # Background QThread workers for scanning and loading series
│   └── sample_data_manager.py # Curated case catalog, manifest tracking & async downloader
├── imaging/
│   ├── coordinate_manager.py  # Patient LPS coordinate transformations & caliper math
│   ├── reslice_manager.py     # Multiplanar Reconstruction (MPR) engine for Axial/Sag/Cor
│   ├── volume_processor.py    # 3D volume container, VTK bridge, procedural phantoms
│   └── window_level.py        # Linear ramp window/level mapping & clinical presets
├── viewers/
│   ├── base_2d_viewer.py      # Base 2D canvas with zoom/pan, HUD, calipers, crosshairs
│   ├── axial_viewer.py        # Axial transverse plane viewport
│   ├── sagittal_viewer.py     # Sagittal longitudinal plane viewport
│   ├── coronal_viewer.py      # Coronal frontal plane viewport
│   ├── volume_3d_viewer.py    # VTK 3D ray-cast volume rendering viewport
│   └── viewport_container.py  # 2x2 resizable grid coordinating synchronization
├── ui/
│   ├── main_window.py         # Main workstation window and shortcuts
│   ├── study_browser.py       # Collapsible left sidebar hierarchy
│   ├── toolbar.py             # Top clinical tool selection and actions
│   ├── status_bar.py          # Status bar with progress and cursor inspector
│   ├── sample_library_dialog.py # Medical Case Library modal browser and downloader
│   ├── welcome_dialog.py      # First-run onboarding dialog
│   └── metadata_dialog.py     # DICOM tag and acquisition parameter table
└── tests/
    ├── test_coordinate_manager.py  # Spatial math and physical distance tests
    ├── test_dicom_loader.py        # Synthetic DICOM slice ordering & loading tests
    ├── test_reslice_and_volume.py  # MPR slice extraction and phantom verification
    ├── test_sample_library.py      # Manifest and case catalog verification
    └── test_ui_workflow.py         # End-to-end UI and 3D rendering integration tests
```

---

## 🚀 Setup & Launch Instructions (Windows)

### 1. Environment Setup
The project uses Python 3.11 with the preconfigured virtual environment in `.venv`:

```powershell
# In PowerShell inside c:\Users\User\Downloads\ide
.\.venv\Scripts\Activate.ps1
```

Or install dependencies into any compatible Python environment:
```powershell
python -m pip install -r requirements.txt
```

### 2. Launching the Workstation
```powershell
.\.venv\Scripts\python.exe main.py
```

### 3. Running Automated Tests
```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 🏥 Built-in Medical Case Library

The application includes both **instant offline procedural phantoms** and **downloadable benchmark clinical datasets**:

1. **CT Head & Cranium (High Contrast)**:
   - *Status*: Ready immediately offline (procedural high-resolution phantom).
   - *Anatomy*: Neurocranium, bone (+1100 HU), brain parenchyma (+38 HU), CSF/ventricles (+15 HU), frontal sinuses (-850 HU), orbital structures.
   - *License*: Open Phantom License.
2. **CT Thorax & Chest Cavity**:
   - *Status*: Ready immediately offline.
   - *Anatomy*: Bilateral lung fields (-720 HU), thoracic spine (+850 HU), sternum (+800 HU), periodic ribs (+950 HU), heart/mediastinum (+45 HU), subcutaneous fat (-90 HU).
   - *License*: Open Phantom License.
3. **3D Slicer Chest CT Sample**:
   - *Source*: Official 3D Slicer benchmark repository.
   - *Modality*: Clinical Chest CT DICOM series.
   - *License*: 3D Slicer BSD-style Open License.
4. **MRI Brain Neuroaxis (T1-Weighted)**:
   - *Source*: 3D Slicer Medical Sample Data.
   - *Modality*: Neurological Magnetic Resonance study.
   - *License*: BSD-style Open License.
5. **CT Skeletal & Bone Extremity**:
   - *Source*: Public Domain Medical Imaging Data.
   - *Modality*: High-resolution bone CT.

### Offline Usage
Datasets downloaded via the **Medical Case Library** are automatically uncompressed, validated, and stored in `./sample_data/<case_id>/`. Downloaded cases persist across restarts and work entirely without an internet connection.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl + O` | Open local DICOM folder |
| `Ctrl + L` | Open Medical Case Library |
| `Ctrl + D` | Quick-load 3D Demo Case |
| `Ctrl + R` | Reset 2D views and 3D camera |
| `W` | Activate Window/Level tool |
| `P` | Activate Pan tool |
| `Z` | Activate Zoom tool |
| `M` | Activate Distance Caliper measurement tool |
| `C` | Activate Crosshair navigation tool |
| `Mouse Wheel` | Slice navigation through active plane |
| `Ctrl + Wheel`| Smooth zoom centered at cursor position |

---

## ⚠️ Medical Device Disclaimer

> [!WARNING]
> This software is intended strictly for medical image visualization, algorithm research, and educational purposes. It is **not** cleared, certified, or approved as a medical device by the FDA, CE, or any other regulatory authority. It must **not** be used as a primary diagnostic system for clinical decision-making.
