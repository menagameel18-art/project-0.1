"""
Pytest configuration and test path setup.
"""
import sys
from pathlib import Path

# Ensure project root directory is in sys.path for test discovery and runners
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
