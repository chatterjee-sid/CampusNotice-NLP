import sys
from pathlib import Path

# src/ modules import each other flat (from preprocess import ...), so put src on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
