#!/usr/bin/env python3
"""Analyze price-level changes and simulate BUY/SELL positions from a CSV."""
import argparse
import math
import sys
import io
from contextlib import redirect_stdout

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure movement through fixed price levels and simulate BUY/SELL positions."
    )
    parser.add_argument("csv_file", help="Input Delta Exchange CSV file")
    parser.add_argument("--start", help='Inclusive start time, e.g. "9/1/2026 12:00:00 AM"')
    parser.add_argument("--end", help='Inclusive end time, e.g. "9/1/2026 11:59:59 PM"')
    parser.add_argument("--base", type=float, help="Base price level; default is the first step level reached")
    parser.add_argument("--step", type=float, default=10.0, help="Distance between levels (default: 10)")
    parser.add_argument("--range", dest="price_range", type=float, default=50.0,
                        help="Allowed distance above and below base; default: 50; use 0 for no range limit")
    parser.add_argument("--events", action="store_true", help="Include every level change and position action in the report")
    parser.add_argument("--output", default="output.txt", help="Output report file (default: output.txt)")
    return parser.parse_args()


def parse_time(value):
    return None if value is None else pd.to_datetime(value, errors="raise")


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def level_for_price(price, base, step):
    """Return the current grid position. A level changes only at a step boundary."""
    index = math.floor((price - base) / step + 1e-12)
    return base + index * step


def crossing_time(t0, t1, p0, p1, level):
    """Estimate when a level was crossed between adjacent samples."""
    if close(p0, p1):
        return t1
    fraction = (level - p0) / (p1 - p0)
    fraction = min(1.0, max(0.0, fraction))
    return t0 + (t1 - t0) * fraction


def first_default_base(df, step):
    """Choose the first grid level reached in the selected range, including a level at the first row."""
    first_price = float(df.iloc[0]["price"])
    first_level = math.ceil(first_price / step - 1e-12) * step
    if close(first_price, first_level):
        return first_level

    for i in range(1, len(df)):
        p0 = float(df.iloc[i - 1]["price"])
        p1 = float(df.iloc[i]["price"])
        low, high = sorted((p0, p1))
        # The first level crossed in this interval, in the direction of travel.
        if p1 > p0:
            level = (math.floor((p0 / step) + 1e-12) + 1) * step
            if level <= p1 + 1e-9:
                return level
        elif p1 < p0:
            level = (math.ceil((p0 / step) - 1e-12) - 1) * step
            if level >= p1 - 1e-9:
                return level
    raise ValueError(
        f"No multiple of {step:g} was crossed in the selected time range. "
        "Specify --base explicitly or choose a wider time range."
    )


def build_events(df, base, step, price_range=50.0):
    """Emit actions only when the next grid level from the current level is crossed.

    The tracked level is stateful: after reaching 2400 from 2410, prices such as
    2399, 2403, and 2405 do not change the tracked level. The next upward event
    requires reaching 2410; the next downward event requires reaching 2390.
    This prevents raw-price band changes from generating premature/repeated
    actions. Large jumps intentionally emit at most one event per CSV interval.
    """
    events = []  # (timestamp, level, direction): +1 up/SELL, -1 down/BUY, 0 base
    lower_range = base - price_range if price_range > 0 else None
    upper_range = base + price_range if price_range > 0 else None
    active = False
    current_level = None
    previous_timestamp = df.iloc[0]["timestamp"]
    previous_price = float(df.iloc[0]["price"])

    if close(previous_price, base):
        events.append((previous_timestamp, base, 0))
        active = True
        current_level = base

    for i in range(1, len(df)):
        timestamp = df.iloc[i]["timestamp"]
        price = float(df.iloc[i]["price"])

        if not active:
            # Start tracking when the base is observed or crossed.
            crossed_base = close(price, base) or (
                min(previous_price, price) < base < max(previous_price, price)
            )
            if crossed_base:
                ts = timestamp if close(price, base) else crossing_time(
                    previous_timestamp, timestamp, previous_price, price, base
                )
                events.append((ts, base, 0))
                active = True
                current_level = base
            previous_timestamp, previous_price = timestamp, price
            continue

        # While the observed price is outside the permitted range, pause trading.
        # Also skip the interval that enters or exits the range: re-entry itself
        # must not create a trade. The tracked level is retained while paused.
        previous_in_range = (
            (lower_range is None or previous_price >= lower_range or close(previous_price, lower_range))
            and (upper_range is None or previous_price <= upper_range or close(previous_price, upper_range))
        )
        current_in_range = (
            (lower_range is None or price >= lower_range or close(price, lower_range))
            and (upper_range is None or price <= upper_range or close(price, upper_range))
        )
        if not previous_in_range or not current_in_range:
            previous_timestamp, previous_price = timestamp, price
            continue

        # Only the adjacent level from the last reached level is actionable.
        # Do not infer extra levels if one sample jumps over several steps.
        if price > previous_price:
            target_level = current_level + step
            crossed = previous_price < target_level and price >= target_level
            direction = 1
        elif price < previous_price:
            target_level = current_level - step
            crossed = previous_price > target_level and price <= target_level
            direction = -1
        else:
            crossed = False
            target_level = current_level
            direction = 0

        if crossed:
            ts = crossing_time(
                previous_timestamp, timestamp, previous_price, price, target_level
            )
            events.append((ts, target_level, direction))
            current_level = target_level

        previous_timestamp, previous_price = timestamp, price

    return events


def simulate_positions(events):
    """Up-level change => SELL; down-level change => BUY; opposite side closes LIFO."""
    open_positions = []
    rows = []
    realized_profit = 0.0

    for i, (timestamp, level, direction) in enumerate(events):
        action = "BASE" if direction == 0 else ("SELL" if direction > 0 else "BUY")
        trade_profit = 0.0

        if action == "BUY":
            match = next((j for j in range(len(open_positions) - 1, -1, -1)
                          if open_positions[j]["side"] == "SELL"), None)
            if match is not None:
                position = open_positions.pop(match)
                trade_profit = position["entry_level"] - level
                action = "BUY (close SELL)"
            else:
                open_positions.append({"side": "BUY", "entry_level": level, "entry_time": timestamp})
                action = "BUY (open)"
        elif action == "SELL":
            match = next((j for j in range(len(open_positions) - 1, -1, -1)
                          if open_positions[j]["side"] == "BUY"), None)
            if match is not None:
                position = open_positions.pop(match)
                trade_profit = level - position["entry_level"]
                action = "SELL (close BUY)"
            else:
                open_positions.append({"side": "SELL", "entry_level": level, "entry_time": timestamp})
                action = "SELL (open)"

        realized_profit += trade_profit
        rows.append({
            "timestamp": timestamp,
            "level": level,
            "action": action,
            "trade_profit": trade_profit,
            "realized_profit": realized_profit,
            "open_buys": sum(p["side"] == "BUY" for p in open_positions),
            "open_sells": sum(p["side"] == "SELL" for p in open_positions),
        })

    return rows, open_positions, realized_profit


def main():
    args = parse_args()
    if args.step <= 0:
        sys.exit("--step must be greater than zero.")
    if args.price_range < 0:
        sys.exit("--range must be zero or greater.")

    try:
        df = pd.read_csv(args.csv_file)
    except Exception as exc:
        sys.exit(f"Could not read CSV: {exc}")

    columns = {str(c).strip().lower(): c for c in df.columns}
    if "timestamp" not in columns or "price" not in columns:
        sys.exit('CSV must contain columns named "timestamp" and "price".')
    df = df.rename(columns={columns["timestamp"]: "timestamp", columns["price"]: "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        sys.exit("No valid timestamp/price rows found.")

    try:
        start, end = parse_time(args.start), parse_time(args.end)
    except Exception as exc:
        sys.exit(f"Could not parse start/end time: {exc}")
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    df = df.reset_index(drop=True)
    if df.empty:
        sys.exit("No CSV rows fall within the selected time range.")

    try:
        base = args.base if args.base is not None else first_default_base(df, args.step)
    except ValueError as exc:
        sys.exit(str(exc))

    events = build_events(df, base, args.step, args.price_range)
    if not events:
        sys.exit(f"Base {base:g} was not reached in the selected time range. Try another --base or time range.")

    total_distance = sum(abs(events[i][1] - events[i - 1][1]) for i in range(1, len(events)))
    sim_rows, open_positions, realized_profit = simulate_positions(events)

    report = io.StringIO()
    with redirect_stdout(report):
        print(f"Rows in selected range : {len(df)}")
        print(f"Selected time range    : {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        print(f"Base                   : {base:g}")
        print(f"Step                   : {args.step:g}")
        print(f"Price range             : ±{args.price_range:g} around base" if args.price_range > 0 else "Price range             : disabled (unbounded)")
        print(f"Effective lower bound   : {base - args.price_range:g}" if args.price_range > 0 else "Effective lower bound   : unbounded")
        print(f"Effective upper bound   : {base + args.price_range:g}" if args.price_range > 0 else "Effective upper bound   : unbounded")
        print(f"Maximum price          : {df['price'].max():g}")
        print(f"Minimum price          : {df['price'].min():g}")
        print(f"Levels reached         : {len(events)}")
        print(f"Total distance         : {total_distance:g}")
        print(f"Realized profit        : {realized_profit:g}")
        print(f"Open BUY positions     : {sum(p['side'] == 'BUY' for p in open_positions)}")
        print(f"Open SELL positions    : {sum(p['side'] == 'SELL' for p in open_positions)}")

        if args.events:
            print("\nPrice-level events and position simulation:")
            header = (f"{'Timestamp':<27} {'Level':>10} {'Action':<20} {'Trade P/L':>12} "
                      f"{'Total P/L':>12} {'Open BUY':>10} {'Open SELL':>11}")
            print(header)
            print("-" * len(header))
            for row in sim_rows:
                ts = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")
                print(f"{ts:<27} {row['level']:>10g} {row['action']:<20} {row['trade_profit']:>12g} "
                      f"{row['realized_profit']:>12g} {row['open_buys']:>10} {row['open_sells']:>11}")

    try:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(report.getvalue())
    except OSError as exc:
        sys.exit(f"Could not write output file '{args.output}': {exc}")

    print(f"Report saved to: {args.output}")
    print(f"Events included: {'yes' if args.events else 'no'} (use --events to include them)")


if __name__ == "__main__":
    main()
