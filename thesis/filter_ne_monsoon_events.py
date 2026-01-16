import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


START_YM = "201710"
END_YM = "202003"


def iter_months(start_ym: str, end_ym: str):
    start = datetime.strptime(start_ym, "%Y%m")
    end = datetime.strptime(end_ym, "%Y%m")
    current = start
    while current <= end:
        yield current.strftime("%Y%m")
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = current.replace(year=year, month=month)


def read_month(path: Path, time_col: str, cols):
    usecols = [time_col] + list(cols)
    df = pd.read_csv(
        path,
        skipinitialspace=True,
        usecols=lambda c: c.strip() in usecols,
        low_memory=False,
    )
    df.columns = [col.strip() for col in df.columns]
    df = df[usecols].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    return df


def load_station(data_dir: Path, prefix: str, months, time_col: str, cols):
    frames = []
    missing = []
    for ym in sorted(months):
        path = data_dir / f"{prefix}{ym}.txt"
        if not path.exists():
            missing.append(path.name)
            continue
        frames.append(read_month(path, time_col, cols))
    if not frames:
        return pd.DataFrame(columns=[time_col] + list(cols)), missing
    df = pd.concat(frames, ignore_index=True)
    return df, missing


def resample_circular_mean(series: pd.Series, freq="10min") -> pd.Series:
    radians = np.deg2rad(series)
    sin_mean = np.sin(radians).resample(freq).mean()
    cos_mean = np.cos(radians).resample(freq).mean()
    angle = np.arctan2(sin_mean, cos_mean)
    deg = np.rad2deg(angle)
    return (deg + 360.0) % 360.0


def circular_mean_deg(series: pd.Series) -> float:
    vals = series.dropna().to_numpy()
    if vals.size == 0:
        return float("nan")
    radians = np.deg2rad(vals)
    sin_mean = float(np.sin(radians).mean())
    cos_mean = float(np.cos(radians).mean())
    angle = np.arctan2(sin_mean, cos_mean)
    deg = np.rad2deg(angle)
    return float((deg + 360.0) % 360.0)


def circular_std_deg(series: pd.Series) -> float:
    vals = series.dropna().to_numpy()
    if vals.size == 0:
        return float("nan")
    radians = np.deg2rad(vals)
    sin_mean = float(np.sin(radians).mean())
    cos_mean = float(np.cos(radians).mean())
    r = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    if r <= 0:
        return float("nan")
    std_rad = np.sqrt(-2.0 * np.log(r))
    return float(np.rad2deg(std_rad))


def wrap_to_180(deg: pd.Series) -> pd.Series:
    return ((deg + 180.0) % 360.0) - 180.0


def valid_fraction(series: pd.Series, freq="10min") -> pd.Series:
    total = series.resample(freq).size()
    valid = series.notna().resample(freq).sum()
    frac = valid / total
    return frac.fillna(0.0)


def build_tp(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.set_index(time_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    speed = df["WS_95"].resample("10min").mean()
    direction = resample_circular_mean(df["WD_95"])
    ws_frac = valid_fraction(df["WS_95"])
    wd_frac = valid_fraction(df["WD_95"])
    frac = pd.concat([ws_frac, wd_frac], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "tp_speed": speed,
            "tp_dir": direction,
            "tp_valid_frac": frac,
        }
    )


def build_bsmi(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.set_index(time_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    speed_raw = df[["WS_100E", "WS_100W"]].mean(axis=1)
    speed = speed_raw.resample("10min").mean()
    direction = resample_circular_mean(df["WD_97"])
    ws_frac = valid_fraction(speed_raw)
    wd_frac = valid_fraction(df["WD_97"])
    frac = pd.concat([ws_frac, wd_frac], axis=1).min(axis=1)
    at = df["AT_95"].resample("10min").mean()
    return pd.DataFrame(
        {
            "bsmi_speed": speed,
            "bsmi_dir": direction,
            "bsmi_valid_frac": frac,
            "bsmi_at": at,
        }
    )


def fill_gaps(mask: pd.Series, max_gap: int) -> pd.Series:
    arr = mask.to_numpy(dtype=bool)
    filled = arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            i += 1
            continue
        start = i
        while i < n and not arr[i]:
            i += 1
        end = i
        gap_len = end - start
        if gap_len <= max_gap and start > 0 and end < n:
            if arr[start - 1] and arr[end]:
                filled[start:end] = True
    return pd.Series(filled, index=mask.index)


def find_true_runs(mask: pd.Series):
    arr = mask.to_numpy(dtype=bool)
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if not arr[i]:
            i += 1
            continue
        start = i
        while i < n and arr[i]:
            i += 1
        end = i
        runs.append((start, end))
    return runs


def plot_event(event_id: int, data: pd.DataFrame, out_dir: Path):
    if data.empty:
        return
    fig, (ax_speed, ax_dir) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_speed.plot(
        data.index, data["tp_speed"], color="tab:orange", linewidth=1.2, label="TP WS"
    )
    ax_speed.plot(
        data.index, data["bsmi_speed"], color="tab:blue", linewidth=1.2, label="BSMI WS"
    )
    ax_speed.set_ylabel("Wind speed (m/s)")
    ax_speed.set_ylim(0, 30)
    ax_speed.grid(True, color="0.9")

    ax_temp = ax_speed.twinx()
    ax_temp.plot(
        data.index, data["bsmi_at"], color="tab:green", linewidth=1.0, label="BSMI AT"
    )
    ax_temp.set_ylabel("BSMI AT (C)")
    ax_temp.set_ylim(0, 25)

    ax_dir.plot(
        data.index, data["tp_dir"], color="tab:orange", linewidth=1.0, label="TP WD"
    )
    ax_dir.plot(
        data.index, data["bsmi_dir"], color="tab:blue", linewidth=1.0, label="BSMI WD"
    )
    ax_dir.set_ylabel("Wind direction (deg)")
    ax_dir.set_ylim(0, 360)
    ax_dir.grid(True, color="0.9")

    ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)

    lines, labels = ax_speed.get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    ax_speed.legend(lines + lines2, labels + labels2, loc="upper left")
    ax_dir.legend(loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"event_{event_id:02d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Filter NE monsoon events and output event table + plots."
    )
    parser.add_argument(
        "--data-dir",
        default="C:/Users/Laura/NCKU/FSSL/cupanemometer/DATA",
        help="Directory containing TP*.txt and BSMI*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="C:/Users/Laura/NCKU/FSSL/cupanemometer/thesis",
        help="Directory to write event table and plots.",
    )
    args = parser.parse_args()

    months = list(iter_months(START_YM, END_YM))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    tp_df, missing_tp = load_station(
        data_dir, "TP", months, "TIMESTAMP", ["WS_95", "WD_95"]
    )
    bsmi_df, missing_bsmi = load_station(
        data_dir,
        "BSMI",
        months,
        "TIMESTAMP",
        ["WS_100E", "WS_100W", "WD_97", "AT_95"],
    )
    if missing_tp:
        print(f"Missing TP files: {', '.join(missing_tp)}")
    if missing_bsmi:
        print(f"Missing BSMI files: {', '.join(missing_bsmi)}")

    tp = build_tp(tp_df, "TIMESTAMP")
    bsmi = build_bsmi(bsmi_df, "TIMESTAMP")
    combined = tp.join(bsmi, how="inner")

    conds = {
        "tp_valid_frac": combined["tp_valid_frac"] >= 0.95,
        "bsmi_valid_frac": combined["bsmi_valid_frac"] >= 0.95,
        "tp_speed": combined["tp_speed"] >= 10.8,
        "bsmi_speed": combined["bsmi_speed"] >= 7.0,
        "tp_dir_ne": (combined["tp_dir"] >= 0.0) & (combined["tp_dir"] <= 45.0),
        "bsmi_dir_ne": (combined["bsmi_dir"] >= 0.0) & (combined["bsmi_dir"] <= 45.0),
        "dir_diff": wrap_to_180(combined["tp_dir"] - combined["bsmi_dir"]).abs() < 30.0,
        "bsmi_at": combined["bsmi_at"] < 20.0,
    }
    mask = pd.concat(conds.values(), axis=1).all(axis=1)

    debug_path = output_dir / "ne_monsoon_filter_debug.csv"
    debug_df = combined.copy()
    for name, cond in conds.items():
        debug_df[f"cond_{name}"] = cond
    debug_df["mask_all"] = mask
    debug_df.to_csv(debug_path, index=True)
    print(f"Wrote {debug_path}")

    total = len(combined)
    print("Condition pass rates:")
    for name, cond in conds.items():
        rate = float(cond.mean()) if total else 0.0
        print(f"  {name}: {rate:.3f}")

    filled = fill_gaps(mask, max_gap=2)
    runs = find_true_runs(filled)

    rows = []
    event_dir = output_dir / "event"
    event_id = 0
    for start_idx, end_idx in runs:
        run_len = end_idx - start_idx
        if run_len * 10 < 72 * 60:
            continue
        run_mask = mask.iloc[start_idx:end_idx]
        satisfied_ratio = float(run_mask.mean()) if run_mask.size else 0.0
        if satisfied_ratio < 0.9:
            continue
        run_data = combined.iloc[start_idx:end_idx]
        tp_std = circular_std_deg(run_data["tp_dir"])
        bsmi_std = circular_std_deg(run_data["bsmi_dir"])
        if np.isnan(tp_std) or np.isnan(bsmi_std):
            continue
        if tp_std >= 10.0 or bsmi_std >= 10.0:
            continue
        event_id += 1
        start_time = run_data.index.min()
        end_time = run_data.index.max()

        rows.append(
            {
                "event_id": f"E{event_id:02d}",
                "start": start_time.strftime("%Y-%m-%d %H:%M"),
                "end": end_time.strftime("%Y-%m-%d %H:%M"),
                "tp_speed_mean": float(run_data["tp_speed"].mean()),
                "tp_dir_mean": circular_mean_deg(run_data["tp_dir"]),
                "tp_dir_std": tp_std,
                "bsmi_speed_mean": float(run_data["bsmi_speed"].mean()),
                "bsmi_dir_mean": circular_mean_deg(run_data["bsmi_dir"]),
                "bsmi_dir_std": bsmi_std,
            }
        )

        plot_event(event_id, run_data, event_dir)

    df = pd.DataFrame(rows)
    float_cols = [
        "tp_speed_mean",
        "tp_dir_mean",
        "tp_dir_std",
        "bsmi_speed_mean",
        "bsmi_dir_mean",
        "bsmi_dir_std",
    ]
    df[float_cols] = df[float_cols].round(2)

    out_csv = output_dir / "ne_monsoon_events.csv"
    df.to_csv(out_csv, index=False, float_format="%.2f")
    table_path = output_dir / "ne_monsoon_events_table.txt"
    table_text = df.to_string(index=False, float_format=lambda v: f"{v:.2f}")
    table_path.write_text(table_text, encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
