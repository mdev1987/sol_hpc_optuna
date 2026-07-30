import typer

from rich import print

from config import CONFIG
from logger import log
from paths import init_directories

app = typer.Typer(add_completion=False)


@app.command()
def run(
    days: int = CONFIG.days,
    trials: int = CONFIG.trials,
):

    init_directories()

    log.info("Replay Optuna")

    print(f"Days: {days}")

    print(f"Trials: {trials}")

    #
    # Stage 2
    #
    # Download Replay
    #


if __name__ == "__main__":
    app()
