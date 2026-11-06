#!/usr/bin/env python3
"""Analyze OHLC price-level changes and simulate BUY/SELL positions."""
import argparse
import io
import math
import sys
from contextlib import redirect_stdout

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure OHLC movement through fixed price levels and simulate BUY/SELL positions."
    )
    parser.add_argument("csv_file", help="Input OHLC CSV file")
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
    return None if value is None else parse_timestamp(value, errors="raise")


def parse_timestamp(value, errors="coerce"):
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    numeric_text = text[:-2] if text.endswith(".0") else text
    if numeric_text.isdecimal() and len(numeric_text) == 16:
        return pd.to_datetime(int(numeric_text), unit="us", errors=errors)
    return pd.to_datetime(value, format="mixed", errors=errors)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def first_default_base(df, step):
    """Choose the first grid level reached by the OHLC data."""
    first_close = float(df.iloc[0]["close"])
    first_level = math.ceil(first_close / step - 1e-12) * step
    if close(first_close, first_level):
        return first_level

    for i in range(1, len(df)):
        previous_close = float(df.iloc[i - 1]["close"])
        high = float(df.iloc[i]["high"])
        low = float(df.iloc[i]["low"])

        up_level = (math.floor(previous_close / step + 1e-12) + 1) * step
        down_level = (math.ceil(previous_close / step - 1e-12) - 1) * step
        crossed_up = high >= up_level or close(high, up_level)
        crossed_down = low <= down_level or close(low, down_level)

        if crossed_up and crossed_down:
            continue
        if crossed_up:
            return up_level
        if crossed_down:
            return down_level

    raise ValueError(
        f"No multiple of {step:g} was crossed in the selected time range. "
        "Specify --base explicitly or choose a wider time range."
    )


def build_events(df, base, step, price_range=50.0):
    """Emit level events using bar lows for BUYs and highs for SELLs.

    Bars that cross the next level in both directions are skipped because OHLC
    data does not establish which extreme occurred first.
    """
    events = []  # (timestamp, level, direction, execution_price)
    lower_range = base - price_range if price_range > 0 else None
    upper_range = base + price_range if price_range > 0 else None
    ambiguous_bars = 0
    active = False
    current_level = None

    first_row = df.iloc[0]
    if float(first_row["low"]) <= base <= float(first_row["high"]):
        events.append((first_row["timestamp"], base, 0, base))
        active = True
        current_level = base

    for i in range(1, len(df)):
        row = df.iloc[i]
        timestamp = row["timestamp"]
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        previous_close = float(df.iloc[i - 1]["close"])

        if not active:
            base_reached = low <= base <= high or (
                min(previous_close, open_price) <= base <= max(previous_close, open_price)
            )
            if not base_reached:
                continue
            events.append((timestamp, base, 0, base))
            active = True
            current_level = base

        up_level = current_level + step
        down_level = current_level - step
        can_sell = (
            (upper_range is None or up_level <= upper_range or close(up_level, upper_range))
            and (high >= up_level or close(high, up_level))
        )
        can_buy = (
            (lower_range is None or down_level >= lower_range or close(down_level, lower_range))
            and (low <= down_level or close(low, down_level))
        )

        if can_sell and can_buy:
            ambiguous_bars += 1
            continue

        if can_sell:
            crossed_levels = []
            level = up_level
            while (
                (upper_range is None or level <= upper_range or close(level, upper_range))
                and (high >= level or close(high, level))
            ):
                crossed_levels.append(level)
                level += step
            events.extend((timestamp, level, 1, high) for level in crossed_levels)
            current_level = crossed_levels[-1]
        elif can_buy:
            crossed_levels = []
            level = down_level
            while (
                (lower_range is None or level >= lower_range or close(level, lower_range))
                and (low <= level or close(low, level))
            ):
                crossed_levels.append(level)
                level -= step
            events.extend((timestamp, level, -1, low) for level in crossed_levels)
            current_level = crossed_levels[-1]

    return events, ambiguous_bars


def simulate_positions(events, base):
    """Up-level change => SELL; down-level change => BUY; opposite side closes LIFO."""
    open_positions = []
    rows = []
    realized_profit = 0.0

    for timestamp, level, direction, execution_price in events:
        action = "BASE" if direction == 0 else ("SELL" if direction > 0 else "BUY")
        trade_profit = 0.0

        if action == "BUY":
            match = next((j for j in range(len(open_positions) - 1, -1, -1)
                          if open_positions[j]["side"] == "SELL"), None)
            if match is not None:
                position = open_positions.pop(match)
                trade_profit = position["entry_price"] - execution_price
                if close(level, base):
                    trade_profit *= 2
                action = "BUY (close SELL)"
            else:
                open_positions.append({"side": "BUY", "entry_price": execution_price, "entry_time": timestamp})
                action = "BUY (open)"
        elif action == "SELL":
            match = next((j for j in range(len(open_positions) - 1, -1, -1)
                          if open_positions[j]["side"] == "BUY"), None)
            if match is not None:
                position = open_positions.pop(match)
                trade_profit = execution_price - position["entry_price"]
                if close(level, base):
                    trade_profit *= 2
                action = "SELL (close BUY)"
            else:
                open_positions.append({"side": "SELL", "entry_price": execution_price, "entry_time": timestamp})
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
    required_columns = ("timestamp", "open", "high", "low", "close")
    missing_columns = [name for name in required_columns if name not in columns]
    if missing_columns:
        sys.exit(
            "CSV must contain columns named "
            f"{', '.join(repr(name) for name in required_columns)}; "
            f"missing: {', '.join(repr(name) for name in missing_columns)}."
        )

    df = df.rename(columns={columns[name]: name for name in required_columns})
    df["timestamp"] = df["timestamp"].map(parse_timestamp)
    for name in ("open", "high", "low", "close"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=list(required_columns)).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        sys.exit("No valid timestamp/OHLC rows found.")

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

    events, ambiguous_bars = build_events(df, base, args.step, args.price_range)
    if not events:
        sys.exit(f"Base {base:g} was not reached in the selected time range. Try another --base or time range.")

    total_distance = sum(abs(events[i][1] - events[i - 1][1]) for i in range(1, len(events)))
    sim_rows, open_positions, realized_profit = simulate_positions(events, base)

    report = io.StringIO()
    with redirect_stdout(report):
        print(f"Rows in selected range : {len(df)}")
        print(f"Selected time range    : {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        print(f"Base                   : {base:g}")
        print(f"Step                   : {args.step:g}")
        print(f"Price range            : ±{args.price_range:g} around base" if args.price_range > 0 else "Price range            : disabled (unbounded)")
        print(f"Effective lower bound  : {base - args.price_range:g}" if args.price_range > 0 else "Effective lower bound  : unbounded")
        print(f"Effective upper bound  : {base + args.price_range:g}" if args.price_range > 0 else "Effective upper bound  : unbounded")
        print(f"Maximum high           : {df['high'].max():g}")
        print(f"Minimum low            : {df['low'].min():g}")
        print(f"Levels reached         : {len(events)}")
        print(f"Ambiguous bars skipped : {ambiguous_bars}")
        print(f"Total distance         : {total_distance:g}")
        print(f"Realized profit        : {realized_profit:g}")
        print(f"Open BUY positions     : {sum(p['side'] == 'BUY' for p in open_positions)}")
        print(f"Open SELL positions    : {sum(p['side'] == 'SELL' for p in open_positions)}")

        if args.events:
            print("\nOHLC price-level events and position simulation:")
            header = (f"{'Timestamp':<27} {'Level':>10} {'Action':<20} {'Fill':>12} "
                      f"{'Trade P/L':>12} {'Total P/L':>12} {'Open BUY':>10} {'Open SELL':>11}")
            print(header)
            print("-" * len(header))
            for event, row in zip(events, sim_rows):
                timestamp, _, _, execution_price = event
                ts = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                print(f"{ts:<27} {row['level']:>10g} {row['action']:<20} {execution_price:>12g} "
                      f"{row['trade_profit']:>12g} {row['realized_profit']:>12g} "
                      f"{row['open_buys']:>10} {row['open_sells']:>11}")

    try:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(report.getvalue())
    except OSError as exc:
        sys.exit(f"Could not write output file '{args.output}': {exc}")

    print(f"Report saved to: {args.output}")
    print(f"Events included: {'yes' if args.events else 'no'} (use --events to include them)")


if __name__ == "__main__":
    main()