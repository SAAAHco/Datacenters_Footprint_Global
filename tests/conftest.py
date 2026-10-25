import os
import sys
from pathlib import Path

# The frozen gates reproduce the regional study, which rejected only the computing load
# through the tower and bisected the outlet air between T2 and 50 C. The production physics
# (dcfootprint.core.tower, v3) is exercised explicitly by gate 9 with legacy=False.
os.environ["DCF_LEGACY_PHYSICS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                     # dcfootprint package
sys.path.insert(0, str(ROOT / "Old_analysis"))    # oracle modules (read only)
