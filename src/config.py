from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic_case_data"
REPORTS_DIR = ROOT_DIR / "reports"
VISUALS_DIR = ROOT_DIR / "visuals"
