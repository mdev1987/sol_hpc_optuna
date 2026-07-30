from pydantic import BaseModel

from constants import OPTUNA_DB


class Config(BaseModel):
    days: int = 3
    download_workers: int = 20
    cpu_workers: int = 30
    trials: int = 2000
    timeout: int = 30
    retries: int = 5
    random_seed: int = 42

    @property
    def storage(self) -> str:
        return f"sqlite:///{OPTUNA_DB}"


CONFIG = Config()
