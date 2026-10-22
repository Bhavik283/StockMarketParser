#!/usr/bin/env python3
"""Combine date-separated price CSV files into one chronological CSV.

Expected columns include `timestamp` and `price`. Files are ordered by the
first data-row timestamp; rows within each file retain their original order.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def parse_timestamp(value: str) -> datetime:
    """Parse timestamps commonly used in the source CSVs."""
    value = value.strip()
    formats = (
        "%m/%d/%Y %I:%M:%S %p",  # 9/1/2026 12:00:37 AM
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {value!r}") from exc


def inspect_csv(path: Path):
    """Return header and first data-row timestamp for a CSV file."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("missing header or file is empty")

        timestamp_column = next(
            (name for name in reader.fieldnames if name.strip().lower() == "timestamp"),
            None,
        )
        if timestamp_column is None:
            raise ValueError("no 'timestamp' column found")

        for row in reader:
            if not row or not any((value or "").strip() for value in row.values()):
                continue
            raw_timestamp = row.get(timestamp_column, "") or ""
            return reader.fieldnames, parse_timestamp(raw_timestamp)

    raise ValueError("contains no data rows")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine CSV files in a folder, ordered by each file's first timestamp."
    )
    parser.add_argument(
        "folder", nargs="?", default=".",
        help="folder containing the source CSV files (default: current folder)",
    )
    parser.add_argument(
        "-o", "--output", default="combined.csv",
        help="output CSV filename or path (default: combined.csv in the input folder)",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: input folder does not exist: {folder}", file=sys.stderr)
        return 1

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = folder / output
    output = output.resolve()

    csv_files = sorted(
        (path for path in folder.glob("*.csv") if path.resolve() != output),
        key=lambda path: path.name.lower(),
    )
    if not csv_files:
        print(f"Error: no source CSV files found in {folder}", file=sys.stderr)
        return 1

    inspected = []
    expected_header = None
    errors = []
    for path in csv_files:
        try:
            header, first_timestamp = inspect_csv(path)
            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise ValueError(
                    f"column headers differ from {csv_files[0].name}: {header!r}"
                )
            inspected.append((first_timestamp, path))
        except (OSError, ValueError, csv.Error) as exc:
            errors.append(f"{path.name}: {exc}")

    if errors:
        print("Error: unable to combine all files:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    inspected.sort(key=lambda item: (item[0], item[1].name.lower()))
    output.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    try:
        with output.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.writer(out_handle)
            writer.writerow(expected_header)

            for first_timestamp, path in inspected:
                with path.open("r", newline="", encoding="utf-8-sig") as in_handle:
                    reader = csv.reader(in_handle)
                    next(reader, None)  # Skip this file's header.
                    file_rows = 0
                    for row in reader:
                        if not row or not any((value or "").strip() for value in row):
                            continue
                        writer.writerow(row)
                        row_count += 1
                        file_rows += 1
                print(
                    f"Added {path.name} (first timestamp: {first_timestamp.isoformat(' ')}, "
                    f"rows: {file_rows})"
                )
    except (OSError, csv.Error) as exc:
        print(f"Error writing output file {output}: {exc}", file=sys.stderr)
        return 1

    print(f"\nCombined {len(inspected)} files and {row_count} rows into:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
