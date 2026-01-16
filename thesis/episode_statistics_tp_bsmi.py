import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPISODES = [
    ("20171122 0100", "20171124 2100"),
    ("20171129 2200", "20171205 2359"),
    ("20171216 0000", "20171221 2000"),
    ("20180109 0000", "20180113 0800"),
    ("20180131 1200", "20180206 2359"),
    ("20181001 1200", "20181004 2359"),
    ("20190112 1200", "20190118 0400"),
    ("20191201 2000", "20191208 2000"),
    ("20200207 0000", "20200209 2359"),
]


def parse_dt(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d %H%M")


def iter_months(start: datetime, end: datetime):
    current = datetime(start.year, start.month, 1)
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


def wrap_to_180(deg: pd.Series) -> pd.Series:
    return ((deg + 180.0) % 360.0) - 180.0


def build_tp(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.set_index(time_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    speed = df["WS_95"].resample("10min").mean()
    direction = resample_circular_mean(df["WD_95"])
    return pd.DataFrame({"tp_speed": speed, "tp_dir": direction})


def build_bsmi(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.set_index(time_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    speed_raw = df[["WS_100E", "WS_100W"]].mean(axis=1)
    speed = speed_raw.resample("10min").mean()
    direction = resample_circular_mean(df["WD_97"])
    return pd.DataFrame({"bsmi_speed": speed, "bsmi_dir": direction})


def compute_metrics(df: pd.DataFrame):
    speed = df[["tp_speed", "bsmi_speed"]].dropna()
    if speed.empty:
        corr = float("nan")
        bias = float("nan")
        rmse = float("nan")
        n_speed = 0
    else:
        corr = speed["tp_speed"].corr(speed["bsmi_speed"])
        diff = speed["tp_speed"] - speed["bsmi_speed"]
        bias = diff.mean()
        rmse = float(np.sqrt((diff ** 2).mean()))
        n_speed = int(speed.shape[0])

    direction = df[["tp_dir", "bsmi_dir"]].dropna()
    if direction.empty:
        dir_diff_mean = float("nan")
        n_dir = 0
        dir_diff = pd.Series(dtype=float)
    else:
        dir_diff = wrap_to_180(direction["tp_dir"] - direction["bsmi_dir"])
        dir_diff_mean = float(dir_diff.mean())
        n_dir = int(direction.shape[0])

    return corr, bias, rmse, dir_diff_mean, n_speed, n_dir, dir_diff


def plot_speed_scatter_grid(speed_list, out_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (label, speed) in zip(axes.flatten(), speed_list):
        speed = speed.dropna()
        if speed.empty:
            ax.set_title(label)
            ax.text(0.05, 0.9, "No data", transform=ax.transAxes)
            ax.grid(True, color="0.9")
            continue
        x = speed["bsmi_speed"].to_numpy()
        y = speed["tp_speed"].to_numpy()
        ax.scatter(
            x,
            y,
            s=12,
            alpha=0.7,
            marker="o",
            linewidths=0,
            edgecolors="none",
        )
        if x.size >= 2:
            a, b = np.polyfit(x, y, 1)
            text = f"y={a:.2f}x+{b:.2f}\n" f"n={x.size}"
        else:
            text = f"y=nan\n" f"n={x.size}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top")
        ax.set_title(label)
        ax.set_xlabel("BSMI WS_100E/WS_100W mean (m/s)")
        ax.set_ylabel("TP WS_95 (m/s)")
        ax.grid(True, color="0.9")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dir_hist_grid(diff_list, out_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    bins = np.arange(-100, 105, 5)
    for ax, (label, diff) in zip(axes.flatten(), diff_list):
        diff = diff.dropna()
        diff = diff[(diff >= -100) & (diff <= 100)]
        if diff.empty:
            ax.set_title(label)
            ax.text(0.05, 0.9, "No data", transform=ax.transAxes)
            ax.set_xlim(-100, 100)
            ax.grid(True, color="0.9")
            continue
        ax.hist(diff, bins=bins, color="tab:blue", alpha=0.7)
        ax.set_xlim(-100, 100)
        ax.set_title(label)
        ax.set_xlabel("Direction diff TP - BSMI (deg)")
        ax.set_ylabel("Count")
        ax.grid(True, color="0.9")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute TP/BSMI wind metrics per episode with 10-min rolling means."
    )
    parser.add_argument(
        "--data-dir",
        default="C:/Users/Laura/NCKU/FSSL/cupanemometer/DATA",
        help="Directory containing TP*.txt and BSMI*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="C:/Users/Laura/NCKU/FSSL/cupanemometer/thesis",
        help="Directory to write episode_statistics.csv and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing plot images.",
    )
    args = parser.parse_args()

    episodes = [(parse_dt(s), parse_dt(e)) for s, e in EPISODES]
    all_months = set()
    for start, end in episodes:
        all_months.update(iter_months(start, end))

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tp_df, missing_tp = load_station(
        data_dir,
        "TP",
        all_months,
        "TIMESTAMP",
        ["WS_95", "WD_95"],
    )
    bsmi_df, missing_bsmi = load_station(
        data_dir,
        "BSMI",
        all_months,
        "TIMESTAMP",
        ["WS_100E", "WS_100W", "WD_97"],
    )

    if missing_tp:
        print(f"Missing TP files: {', '.join(missing_tp)}")
    if missing_bsmi:
        print(f"Missing BSMI files: {', '.join(missing_bsmi)}")

    tp_roll = build_tp(tp_df, "TIMESTAMP")
    bsmi_roll = build_bsmi(bsmi_df, "TIMESTAMP")
    combined = tp_roll.join(bsmi_roll, how="inner")

    rows = []
    speed_plots = []
    dir_plots = []
    for idx, (start, end) in enumerate(episodes, start=1):
        subset = combined.loc[start:end]
        corr, bias, rmse, dir_diff_mean, n_speed, n_dir, dir_diff = compute_metrics(subset)
        label = f"E{idx:02d}"
        rows.append(
            {
                "episode": label,
                "start": start.strftime("%Y-%m-%d %H:%M"),
                "end": end.strftime("%Y-%m-%d %H:%M"),
                "wind_speed_corr": corr,
                "wind_speed_bias": bias,
                "wind_speed_rmse": rmse,
                "wind_dir_diff_mean": dir_diff_mean,
                "n_speed": n_speed,
                "n_dir": n_dir,
            }
        )

        if not args.no_plots:
            speed_plots.append((label, subset[["tp_speed", "bsmi_speed"]]))
            dir_plots.append((label, dir_diff))

    df = pd.DataFrame(rows)
    float_cols = [
        "wind_speed_corr",
        "wind_speed_bias",
        "wind_speed_rmse",
        "wind_dir_diff_mean",
    ]
    df[float_cols] = df[float_cols].round(2)
    out_csv = output_dir / "episode_statistics.csv"
    df.to_csv(out_csv, index=False, float_format="%.2f")
    table_path = output_dir / "episode_statistics_table.txt"
    table_text = df.to_string(index=False, float_format=lambda v: f"{v:.2f}")
    table_path.write_text(table_text, encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {table_path}")

    if not args.no_plots:
        plot_speed_scatter_grid(
            speed_plots,
            output_dir / "episode_windspeed_scatter_3x3.png",
        )
        plot_dir_hist_grid(
            dir_plots,
            output_dir / "episode_winddir_diff_hist_3x3.png",
        )


if __name__ == "__main__":
    main()
