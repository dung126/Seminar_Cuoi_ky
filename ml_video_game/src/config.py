from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)