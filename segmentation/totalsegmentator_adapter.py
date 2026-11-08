'''Adapter for TotalSegmentator CLI

This lightweight wrapper runs the official TotalSegmentator tool in a separate
process, captures the generated NIfTI mask file paths and returns them as a list.
It is deliberately minimal – UI code handles progress feedback.
'''

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def run_totalsegmentator(
    input_nii: str,
    output_dir: str,
    task: str = "all",
    structures: Optional[List[str]] = None,
) -> List[str]:
    """Run TotalSegmentator on a NIfTI volume.

    Parameters
    ----------
    input_nii: str
        Path to the input NIfTI file (CT volume).
    output_dir: str
        Directory where the masks will be written. It will be created if missing.
    task: str, default "all"
        Which predefined task to run (e.g. "total", "lung_vessels", …). ``"all"``
        yields the full set of anatomical structures.
    structures: list[str] | None
        Optional explicit list of structures to keep. If provided, ``--ml`` is
        passed to TotalSegmentator to limit output.

    Returns
    -------
    List[str]
        Absolute paths of the generated ``*_seg.nii.gz`` mask files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "totalsegmentator",
        "-i",
        str(input_nii),
        "-o",
        str(output_path),
        "--task",
        task,
    ]
    if structures:
        cmd.extend(["--ml", ",".join(structures)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator failed (exit {result.returncode}):\n{result.stderr}"
        )

    mask_files = sorted(str(p) for p in output_path.glob("*_seg.nii.gz"))
    if not mask_files:
        raise RuntimeError(
            f"TotalSegmentator completed but no mask files were found in {output_path}"
        )
    return mask_files

