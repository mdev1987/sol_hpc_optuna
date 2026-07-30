from pydantic import BaseModel


class Config(BaseModel):
    days: int = 7

    download_workers: int = 20

    cpu_workers: int = 30

    trials: int = 3000

    timeout: int = 30

    retries: int = 5

    random_seed: int = 42

    storage: str = "sqlite:///optuna.db"


CONFIG = Config()
