from pathlib import Path

ROOT = Path(__file__).parent

DOWNLOAD_DIR = ROOT / "downloads"
PARQUET_DIR = ROOT / "parquet"
CACHE_DIR = ROOT / "cache"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

BASE_URL = "https://replay.pumpapi.io"

DEFAULT_DAYS = 7

DOWNLOAD_WORKERS = 20
CPU_WORKERS = 30

TIMEOUT = 30
RETRIES = 5

USER_AGENT = "ReplayOptuna/1.0"

OPTUNA_DB = ROOT / "optuna.db"
