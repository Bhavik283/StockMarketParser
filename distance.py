#!/usr/bin/env python3
import argparse
import math
import sys
from datetime import timedelta

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure price distance travelled through fixed price levels."
    )
    parser.add_argument("csv_file", help="Delta Exchange CSV file")
    parser.add_argument("--start", help='Inclusive start time, e.g. "9/1/2026 12:00:00 AM"')
    parser.add_argument("--end", help='Inclusive end time, e.g. "9/1/2026 11:59:59 PM"')
    parser.add_argument("--base", type=float, help="Starting price level; default is first step level crossed")
    parser.add_argument("--step", type=float, default=10.0, help="Distance between levels (default: 10)")
    parser.add_argument("--events", action="store_true", help="Print each crossed level and estimated timestamp")
    return parser.parse_args()


def parse_time(value):
    if value is None:
        return None
    result = pd.to_datetime(value, errors="raise")
    return result


def level_crossings(p0, p1, base, step):
    """Return grid levels crossed from p0 to p1, excluding p0 and including p1."""
    if p1 > p0:
        first_k = math.floor((p0 - base) / step) + 1
        last_k = math.floor((p1 - base) / step)
        return [base + k * step for k in range(first_k, last_k + 1)]
    if p1 < p0:
        first_k = math.ceil((p0 - base) / step) - 1
        last_k = math.ceil((p1 - base) / step)
        return [base + k * step for k in range(first_k, last_k - 1, -1)]
    return []


def crossing_time(t0, t1, p0, p1, level):
    """Estimate crossing time by linear interpolation between adjacent CSV rows."""
    if p0 == p1:
        return t1
    fraction = (level - p0) / (p1 - p0)
    return t0 + (t1 - t0) * fraction


def first_default_base(rows, step):
    """Find the first multiple of step touched/crossed during the selected range."""
    first_price = float(rows.iloc[0]["price"])
    nearest_k = round(first_price / step)
    if math.isclose(first_price, nearest_k * step, abs_tol=1e-9):
        return nearest_k * step

    for i in range(1, len(rows)):
        p0 = float(rows.iloc[i - 1]["price"])
        p1 = float(rows.iloc[i]["price"])
        levels = level_crossings(p0, p1, 0.0, step)
        if levels:
            return levels[0]
    raise ValueError(
        f"No multiple of {step:g} was crossed in the selected time range. "
        "Specify --base explicitly or use a wider time range."
    )


def main():
    args = parse_args()
    if args.step <= 0:
        sys.exit("--step must be greater than zero.")

    df = pd.read_csv(args.csv_file)
    # Match column names case-insensitively and ignore surrounding spaces.
    columns = {str(c).strip().lower(): c for c in df.columns}
    if "timestamp" not in columns or "price" not in columns:
        sys.exit('CSV must contain columns named "timestamp" and "price".')

    df = df.rename(columns={columns["timestamp"]: "timestamp", columns["price"]: "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        sys.exit("No valid timestamp/price rows found.")

    start = parse_time(args.start)
    end = parse_time(args.end)
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    df = df.reset_index(drop=True)

    if df.empty:
        sys.exit("No CSV rows fall within the selected time range.")
    if len(df) < 2 and args.base is None:
        sys.exit("At least two rows are needed to find the default base. Specify --base.")

    base = args.base if args.base is not None else first_default_base(df, args.step)

    # Wait until base is first reached/crossed, then count each further grid-level crossing.
    events = []
    active = False

    first_price = float(df.iloc[0]["price"])
    first_time = df.iloc[0]["timestamp"]
    if math.isclose(first_price, base, abs_tol=1e-9):
        events.append((first_time, base))
        active = True

    for i in range(1, len(df)):
        t0, t1 = df.iloc[i - 1]["timestamp"], df.iloc[i]["timestamp"]
        p0, p1 = float(df.iloc[i - 1]["price"]), float(df.iloc[i]["price"])

        if not active:
            # Base must lie between the two samples, inclusive.
            if min(p0, p1) <= base <= max(p0, p1) and p0 != p1:
                t_base = crossing_time(t0, t1, p0, p1, base)
                events.append((t_base, base))
                active = True
                # Continue along the same segment after the base crossing.
                remaining_levels = [
                    level for level in level_crossings(base, p1, base, args.step)
                    if not math.isclose(level, base, abs_tol=1e-9)
                ]
                for level in remaining_levels:
                    events.append((crossing_time(t_base, t1, base, p1, level), level))
            elif math.isclose(p1, base, abs_tol=1e-9):
                events.append((t1, base))
                active = True
            continue

        for level in level_crossings(p0, p1, base, args.step):
            # Avoid duplicating the base if a segment starts at it.
            event_time = crossing_time(t0, t1, p0, p1, level)
            if events and math.isclose(level, events[-1][1], abs_tol=1e-9):
                continue
            events.append((event_time, level))

    if not events:
        sys.exit(
            f"Base {base:g} was not reached in the selected time range. "
            "Try a different --base or time range."
        )

    total_distance = sum(abs(events[i][1] - events[i - 1][1]) for i in range(1, len(events)))

    print(f"Rows in selected range : {len(df)}")
    print(f"Selected time range    : {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    print(f"Base                   : {base:g}")
    print(f"Step                   : {args.step:g}")
    print(f"Levels reached         : {len(events)}")
    print(f"Total distance         : {total_distance:g}")

    if args.events:
        print("\nEstimated level crossings:")
        print(f"{'Timestamp':<30} {'Level':>12} {'Distance from previous':>24}")
        print("-" * 68)
        for i, (timestamp, level) in enumerate(events):
            distance = 0 if i == 0 else abs(level - events[i - 1][1])
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'):<30} {level:>12g} {distance:>24g}")


if __name__ == "__main__":
    main()
