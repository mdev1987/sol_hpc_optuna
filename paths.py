from constants import *


def init_directories():

    for d in (
        DOWNLOAD_DIR,
        PARQUET_DIR,
        CACHE_DIR,
        REPORT_DIR,
        LOG_DIR,
        MODEL_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
