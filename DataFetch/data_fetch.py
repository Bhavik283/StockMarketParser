
import csv
import calendar
import time
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path

import requests

CONFIG_FILE = Path("config.txt")
OUTPUT_DIR = Path("output")

BASE_URL = "https://api.kite.trade/instruments/historical"
INTERVAL = "minute"

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", "oi"
]


def read_config():
    config = {}

    for line_number, line in enumerate(
        CONFIG_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            raise ValueError(
                f"Invalid config format on line {line_number}: {line}"
            )

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key in config:
            raise ValueError(f"Duplicate configuration key: {key}")

        config[key] = value

    required = {
        "instrumentID", "startDate", "endDate", "token1", "token2"
    }

    missing = required - config.keys()
    unknown = config.keys() - required

    if missing:
        raise ValueError(f"Missing config parameters: {sorted(missing)}")

    if unknown:
        raise ValueError(f"Unknown config parameters: {sorted(unknown)}")

    if not all(config[key] for key in required):
        raise ValueError("All five configuration values must be non-empty.")

    start_date = date.fromisoformat(config["startDate"])
    end_date = date.fromisoformat(config["endDate"])

    if start_date > end_date:
        raise ValueError("startDate must not be after endDate.")

    return (
        config["instrumentID"],
        start_date,
        end_date,
        config["token1"],
        config["token2"],
    )


def monthly_ranges(start_date, end_date):
    current = start_date

    while current <= end_date:
        last_day = calendar.monthrange(
            current.year, current.month
        )[1]

        month_end = min(
            date(current.year, current.month, last_day),
            end_date,
        )

        yield current, month_end
        current = month_end + timedelta(days=1)


def download_month(session, instrument_id, start_date, end_date):
    url = f"{BASE_URL}/{instrument_id}/{INTERVAL}"

    params = {
        "from": datetime.combine(
            start_date, dt_time.min
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "to": datetime.combine(
            end_date, dt_time.max
        ).strftime("%Y-%m-%d %H:%M:%S"),
    }

    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()

    result = response.json()

    if result.get("status") != "success":
        raise RuntimeError(f"Kite API error: {result}")

    return result.get("data", {}).get("candles", [])


def save_csv(filename, candles):
    has_oi = any(len(candle) > 6 for candle in candles)
    columns = COLUMNS if has_oi else COLUMNS[:-1]

    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        for candle in candles:
            writer.writerow(candle)


def main():
    instrument_id, start_date, end_date, token1, token2 = read_config()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "X-Kite-Version": "3",
        "Authorization": f"token {token1}:{token2}",
    }

    with requests.Session() as session:
        session.headers.update(headers)

        ranges = list(monthly_ranges(start_date, end_date))
        total = len(ranges)

        for index, (month_start, month_end) in enumerate(ranges, start=1):
            filename = OUTPUT_DIR / f"{month_start:%Y-%m}.csv"

            print(
                f"[{index}/{total}] Downloading "
                f"{month_start} to {month_end}..."
            )

            try:
                candles = download_month(
                    session, instrument_id, month_start, month_end
                )

                save_csv(filename, candles)

                print(
                    f"Saved {filename}: {len(candles)} candles"
                )

            except (requests.RequestException, ValueError, RuntimeError) as exc:
                print(f"Failed for {month_start:%Y-%m}: {exc}")

            time.sleep(0.5)


if __name__ == "__main__":
    main()
