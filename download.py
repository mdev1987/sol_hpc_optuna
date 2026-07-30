"""
download.py

High-performance asynchronous replay downloader.

Features
--------
* HTTP/2
* asyncio
* concurrent downloads
* retry with exponential backoff
* atomic writes
* skip existing files
* graceful cancellation
* rich progress bar
* resume-safe
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import httpx

from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from constants import (
    BASE_URL,
    DOWNLOAD_DIR,
    DOWNLOAD_WORKERS,
    RETRIES,
    TIMEOUT,
    USER_AGENT,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "*/*",
    "Connection": "keep-alive",
}

LIMITS = httpx.Limits(
    max_connections=DOWNLOAD_WORKERS,
    max_keepalive_connections=DOWNLOAD_WORKERS,
)

CLIENT_TIMEOUT = httpx.Timeout(
    connect=TIMEOUT,
    read=TIMEOUT,
    write=TIMEOUT,
    pool=TIMEOUT,
)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------


@dataclass(slots=True)
class ReplayFile:
    day: date
    hour: int

    @property
    def url(self) -> str:
        return (
            f"{BASE_URL}/"
            f"{self.day.year:04d}/"
            f"{self.day.month:02d}/"
            f"{self.day.day:02d}/"
            f"{self.hour:02d}.jsonl.zst"
        )

    @property
    def directory(self) -> Path:
        return (
            DOWNLOAD_DIR
            / f"{self.day.year:04d}"
            / f"{self.day.month:02d}"
            / f"{self.day.day:02d}"
        )

    @property
    def filename(self) -> str:
        return f"{self.hour:02d}.jsonl.zst"

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    @property
    def temporary(self) -> Path:
        return self.directory / f"{self.filename}.part"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def previous_days(days: int) -> list[date]:
    """
    Previous N UTC days ending yesterday.
    """

    yesterday = datetime.now(UTC).date() - timedelta(days=1)

    return [yesterday - timedelta(days=i) for i in reversed(range(days))]


def build_replay_list(days: int) -> list[ReplayFile]:
    files: list[ReplayFile] = []

    for day in previous_days(days):
        for hour in range(24):
            files.append(
                ReplayFile(
                    day=day,
                    hour=hour,
                )
            )

    return files


# ------------------------------------------------------------
# Downloader
# ------------------------------------------------------------


class ReplayDownloader:
    def __init__(
        self,
        workers: int = DOWNLOAD_WORKERS,
    ):

        self.workers = workers

        self.queue: asyncio.Queue[ReplayFile] = asyncio.Queue()

        self.client: httpx.AsyncClient | None = None

        self.progress: Progress | None = None

        self.task_id: int | None = None

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    async def _create_client(self) -> httpx.AsyncClient:

        return httpx.AsyncClient(
            http2=True,
            headers=HEADERS,
            limits=LIMITS,
            timeout=CLIENT_TIMEOUT,
            follow_redirects=True,
        )

    async def _sleep(self, retry: int) -> None:
        """
        Exponential backoff.
        """

        await asyncio.sleep(min(2**retry, 30))

    def _advance(self) -> None:

        if self.progress is not None and self.task_id is not None:
            self.progress.advance(self.task_id)

    # ------------------------------------------------------------
    # Download
    # ------------------------------------------------------------

    async def _download(
        self,
        replay: ReplayFile,
    ) -> None:

        assert self.client is not None

        if replay.path.exists():
            self._advance()
            return

        replay.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if replay.temporary.exists():
            replay.temporary.unlink()

        last_error: Exception | None = None

        for retry in range(RETRIES):
            try:
                async with self.client.stream(
                    "GET",
                    replay.url,
                ) as response:
                    response.raise_for_status()

                    with replay.temporary.open("wb") as fp:
                        async for chunk in response.aiter_bytes():
                            fp.write(chunk)

                os.replace(
                    replay.temporary,
                    replay.path,
                )

                self._advance()

                return

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                last_error = exc

                if replay.temporary.exists():
                    replay.temporary.unlink()

                if retry + 1 != RETRIES:
                    await self._sleep(retry)

        raise RuntimeError(f"Failed downloading {replay.url}") from last_error

    # ------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------

    async def _worker(
        self,
        worker_id: int,
    ) -> None:

        while True:
            replay = await self.queue.get()

            try:
                await self._download(replay)

            finally:
                self.queue.task_done()

    # ------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------

    async def _fill_queue(
        self,
        files: list[ReplayFile],
    ) -> None:

        for replay in files:
            await self.queue.put(replay)

    # ------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------

    def _progress(self) -> Progress:

        return Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    async def download(
        self,
        days: int,
    ) -> None:
        """
        Download replay archives for the previous N UTC days.

        Parameters
        ----------
        days:
            Number of previous UTC days ending yesterday.
        """

        files = build_replay_list(days)

        async with await self._create_client() as client:
            self.client = client

            with self._progress() as progress:
                self.progress = progress

                self.task_id = progress.add_task(
                    "Replay",
                    total=len(files),
                )

                #
                # Fill queue
                #
                await self._fill_queue(files)

                #
                # Start workers
                #
                workers = [
                    asyncio.create_task(
                        self._worker(i),
                        name=f"worker-{i}",
                    )
                    for i in range(self.workers)
                ]

                #
                # Wait until queue is empty
                #
                await self.queue.join()

                #
                # Stop workers
                #
                for worker in workers:
                    worker.cancel()

                await asyncio.gather(
                    *workers,
                    return_exceptions=True,
                )

        self.client = None
        self.progress = None
        self.task_id = None

    # ------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------

    @classmethod
    async def run(
        cls,
        days: int,
        workers: int = DOWNLOAD_WORKERS,
    ) -> None:

        downloader = cls(
            workers=workers,
        )

        await downloader.download(days)
        # ------------------------------------------------------------


# Statistics
# ------------------------------------------------------------

from dataclasses import dataclass
import asyncio
import signal
import time


@dataclass(slots=True)
class DownloadStats:
    started: float = 0.0
    finished: float = 0.0
    total_files: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def elapsed(self) -> float:
        if self.finished == 0:
            return time.perf_counter() - self.started
        return self.finished - self.started

    @property
    def success(self) -> int:
        return self.downloaded + self.skipped


# ------------------------------------------------------------
# Graceful Shutdown
# ------------------------------------------------------------


class ShutdownHandler:
    def __init__(self):

        self.stop = asyncio.Event()

    def install(self):

        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    self.stop.set,
                )
            except NotImplementedError:
                # Windows
                signal.signal(
                    sig,
                    lambda *_: self.stop.set(),
                )


# ------------------------------------------------------------
# Standalone Runner
# ------------------------------------------------------------


async def _main():

    import argparse

    parser = argparse.ArgumentParser(
        description="PumpAPI Replay Downloader",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
    )

    args = parser.parse_args()

    handler = ShutdownHandler()

    handler.install()

    stats = DownloadStats()

    stats.started = time.perf_counter()

    files = build_replay_list(args.days)

    stats.total_files = len(files)

    print(f"Downloading {stats.total_files} replay files...")
    print(f"Workers : {args.workers}")
    print(f"Output  : {DOWNLOAD_DIR}")
    print()

    task = asyncio.create_task(
        download_replay(
            days=args.days,
            workers=args.workers,
        )
    )

    done, pending = await asyncio.wait(
        {
            task,
            asyncio.create_task(handler.stop.wait()),
        },
        return_when=asyncio.FIRST_COMPLETED,
    )

    if handler.stop.is_set():
        print()
        print("Stopping...")

        task.cancel()

        await asyncio.gather(
            task,
            return_exceptions=True,
        )

        return

    await task

    stats.finished = time.perf_counter()

    print()
    print("=" * 60)
    print("Download Finished")
    print("=" * 60)
    print(f"Elapsed : {stats.elapsed:.2f} seconds")
    print(f"Files   : {stats.total_files}")
    print("=" * 60)


def main():

    asyncio.run(_main())


if __name__ == "__main__":
    main()


async def download_replay(
    days: int,
    workers: int = DOWNLOAD_WORKERS,
) -> None:
    """
    Convenience function.

    Example
    -------
    await download_replay(7)
    """

    await ReplayDownloader.run(
        days=days,
        workers=workers,
    )
