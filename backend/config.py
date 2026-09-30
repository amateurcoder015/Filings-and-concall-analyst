import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FILINGS_DATA_DIR", ROOT / "data" / "raw" / "infosys"))
DB_PATH = Path(os.environ.get("FILINGS_DB", ROOT / "data" / "index" / "index.db"))
MODEL = os.environ.get("FILINGS_MODEL", "claude-sonnet-5-5")
