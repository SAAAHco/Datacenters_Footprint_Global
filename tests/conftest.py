import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                     # dcfootprint package
sys.path.insert(0, str(ROOT / "Old_analysis"))    # oracle modules (read only)
