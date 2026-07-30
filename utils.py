from datetime import UTC
from datetime import datetime
from datetime import timedelta


def yesterday():

    return datetime.now(UTC).date() - timedelta(days=1)


def daterange(days):

    end = yesterday()

    start = end - timedelta(days=days - 1)

    current = start

    while current <= end:
        yield current

        current += timedelta(days=1)
