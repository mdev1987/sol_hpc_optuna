from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TextColumn,
)

from constants import (
    BASE_URL,
    DOWNLOAD_DIR,
    DOWNLOAD_WORKERS,
    RETRIES,
    TIMEOUT,
    USER_AGENT,
)

HEADERS = {
    "User-Agent": USER_AGENT,
}


def replay_url(day: date, hour: int) -> str:
    return (
        f"{BASE_URL}/{day.year:04d}/{day.month:02d}/{day.day:02d}/{hour:02d}.jsonl.zst"
    )


def replay_path(day: date, hour: int) -> Path:
    return (
        DOWNLOAD_DIR
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / f"{day.day:02d}"
        / f"{hour:02d}.jsonl.zst"
    )


def previous_days(days: int) -> list[date]:
    yesterday = datetime.now(UTC).date() - timedelta(days=1)

    return [yesterday - timedelta(days=i) for i in reversed(range(days))]


class Downloader:
    def __init__(self, workers: int = DOWNLOAD_WORKERS):

        self.sem = asyncio.Semaphore(workers)

    async def _download(
        self,
        client: httpx.AsyncClient,
        url: str,
        path: Path,
        progress: Progress,
        task: int,
    ):

        if path.exists():
            progress.advance(task)
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        async with self.sem:
            for retry in range(RETRIES):
                try:
                    async with client.stream("GET", url) as r:
                        r.raise_for_status()

                        with path.open("wb") as f:
                            async for chunk in r.aiter_bytes():
                                f.write(chunk)

                    progress.advance(task)

                    return

                except Exception:
                    if retry + 1 == RETRIES:
                        raise

                    await asyncio.sleep(2**retry)

    async def download_week(self, days: int):

        files = []

        for day in previous_days(days):
            for hour in range(24):
                files.append(
                    (
                        replay_url(day, hour),
                        replay_path(day, hour),
                    )
                )

        timeout = httpx.Timeout(TIMEOUT)

        limits = httpx.Limits(
            max_connections=DOWNLOAD_WORKERS,
            max_keepalive_connections=DOWNLOAD_WORKERS,
        )

        async with httpx.AsyncClient(
            http2=True,
            timeout=timeout,
            headers=HEADERS,
            limits=limits,
        ) as client:
            with Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "Replay",
                    total=len(files),
                )

                await asyncio.gather(
                    *[
                        self._download(
                            client,
                            url,
                            path,
                            progress,
                            task,
                        )
                        for url, path in files
                    ]
                )
