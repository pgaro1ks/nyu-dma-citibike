import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    FIGURES_DIR,
    STUDY_AREA,
    MORNING_PEAK,
    TRIP_MONTHS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def load_study_stations() -> pd.DataFrame:
    path = DATA_PROCESSED / "stations_study_area.parquet"
    if not path.exists():
        print("[error] Run 02_station_meta.py first")
        sys.exit(1)
    return pd.read_parquet(path)


def load_trips() -> pd.DataFrame:
    csv_files = sorted(DATA_RAW.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("JC-")]

    if not csv_files:
        print("[error] No trip CSVs found. Run 01_download_trips.py first")
        sys.exit(1)

    print(f"[load] {len(csv_files)} trip CSV file(s)")
    frames = []
    for f in csv_files:
        print(f"  Reading {f.name}...")
        df = pd.read_csv(
            f,
            usecols=[
                "started_at",
                "ended_at",
                "start_station_id",
                "end_station_id",
                "start_lat",
                "start_lng",
                "end_lat",
                "end_lng",
                "rideable_type",
                "member_casual",
            ],
            parse_dates=["started_at", "ended_at"],
            dtype={"start_station_id": str, "end_station_id": str},
        )
        frames.append(df)
        print(f"    {len(df):,} trips")

    trips = pd.concat(frames, ignore_index=True)
    print(f"[total] {len(trips):,} trips loaded")
    return trips


def filter_study_area_trips(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    study_ids = set(stations["station_id"].values)

    mask = (
        trips["start_station_id"].isin(study_ids)
        | trips["end_station_id"].isin(study_ids)
    )
    filtered = trips[mask].copy()
    print(f"[filter] {len(filtered):,} trips touch study area ({len(trips):,} total)")
    return filtered


def compute_hourly_flows(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    study_ids = set(stations["station_id"].values)

    departures = trips[trips["start_station_id"].isin(study_ids)].copy()
    departures["station_id"] = departures["start_station_id"]
    departures["hour"] = departures["started_at"].dt.hour
    departures["date"] = departures["started_at"].dt.date
    departures["flow_type"] = "departure"

    arrivals = trips[trips["end_station_id"].isin(study_ids)].copy()
    arrivals["station_id"] = arrivals["end_station_id"]
    arrivals["hour"] = arrivals["ended_at"].dt.hour
    arrivals["date"] = arrivals["ended_at"].dt.date
    arrivals["flow_type"] = "arrival"

    dep_counts = (
        departures.groupby(["station_id", "date", "hour"])
        .size()
        .reset_index(name="departures")
    )
    arr_counts = (
        arrivals.groupby(["station_id", "date", "hour"])
        .size()
        .reset_index(name="arrivals")
    )

    flows = dep_counts.merge(
        arr_counts, on=["station_id", "date", "hour"], how="outer"
    ).fillna(0)

    flows["departures"] = flows["departures"].astype(int)
    flows["arrivals"] = flows["arrivals"].astype(int)
    flows["net_flow"] = flows["arrivals"] - flows["departures"]

    print(f"[flows] {len(flows):,} station-date-hour records")
    return flows


def fit_demand_distributions(flows: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    peak_start = MORNING_PEAK["start_hour"]
    peak_end = MORNING_PEAK["end_hour"]

    morning = flows[(flows["hour"] >= peak_start) & (flows["hour"] < peak_end)]

    morning_daily = (
        morning.groupby(["station_id", "date"])
        .agg(
            morning_departures=("departures", "sum"),
            morning_arrivals=("arrivals", "sum"),
            morning_net=("net_flow", "sum"),
        )
        .reset_index()
    )

    station_stats = (
        morning_daily.groupby("station_id")
        .agg(
            mean_departures=("morning_departures", "mean"),
            std_departures=("morning_departures", "std"),
            mean_arrivals=("morning_arrivals", "mean"),
            std_arrivals=("morning_arrivals", "std"),
            mean_net=("morning_net", "mean"),
            std_net=("morning_net", "std"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )

    station_stats = station_stats.merge(
        stations[["station_id", "name", "capacity", "lat", "lon"]],
        on="station_id",
        how="left",
    )

    station_stats["departure_rate_per_hour"] = station_stats["mean_departures"] / (
        peak_end - peak_start
    )
    station_stats["arrival_rate_per_hour"] = station_stats["mean_arrivals"] / (
        peak_end - peak_start
    )

    station_stats["poisson_lambda_dep"] = station_stats["departure_rate_per_hour"]
    station_stats["poisson_lambda_arr"] = station_stats["arrival_rate_per_hour"]

    station_stats["cv_departures"] = (
        station_stats["std_departures"] / station_stats["mean_departures"]
    ).fillna(0)

    station_stats["station_type"] = np.where(
        station_stats["mean_net"] < -2,
        "net_supplier",
        np.where(station_stats["mean_net"] > 2, "net_receiver", "balanced"),
    )

    print(f"\n[distributions] Fitted for {len(station_stats)} stations")
    print(f"  Net suppliers (more departures): {(station_stats['station_type'] == 'net_supplier').sum()}")
    print(f"  Net receivers (more arrivals):   {(station_stats['station_type'] == 'net_receiver').sum()}")
    print(f"  Balanced:                        {(station_stats['station_type'] == 'balanced').sum()}")

    return station_stats


def plot_system_overview(flows: pd.DataFrame, station_stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Citi Bike Study Area — Demand Profile (Sep–Nov 2024)", fontsize=14, fontweight="bold")

    hourly = flows.groupby("hour").agg(
        departures=("departures", "mean"),
        arrivals=("arrivals", "mean"),
    )
    axes[0, 0].plot(hourly.index, hourly["departures"], "o-", label="Departures", color="#E24B4A")
    axes[0, 0].plot(hourly.index, hourly["arrivals"], "s-", label="Arrivals", color="#378ADD")
    axes[0, 0].axvspan(7, 9, alpha=0.15, color="orange", label="Morning peak")
    axes[0, 0].axvspan(17, 19, alpha=0.15, color="purple", label="Evening peak")
    axes[0, 0].set_xlabel("Hour of day")
    axes[0, 0].set_ylabel("Avg trips (all study stations)")
    axes[0, 0].set_title("Hourly demand pattern")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xticks(range(0, 24, 2))

    colors = {"net_supplier": "#E24B4A", "net_receiver": "#378ADD", "balanced": "#888780"}
    for stype, color in colors.items():
        subset = station_stats[station_stats["station_type"] == stype]
        axes[0, 1].scatter(
            subset["lon"], subset["lat"],
            s=subset["capacity"] * 3,
            c=color, alpha=0.6, label=stype, edgecolors="white", linewidth=0.5,
        )
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")
    axes[0, 1].set_title("Station types (morning peak)")
    axes[0, 1].legend(fontsize=8)

    top_dep = station_stats.nlargest(15, "mean_departures")
    axes[1, 0].barh(
        top_dep["name"].str[:30], top_dep["mean_departures"], color="#E24B4A", alpha=0.8
    )
    axes[1, 0].set_xlabel("Avg morning departures")
    axes[1, 0].set_title("Top 15 departure stations (7–9 AM)")
    axes[1, 0].invert_yaxis()

    axes[1, 1].hist(
        station_stats["mean_net"], bins=30, color="#378ADD", alpha=0.7, edgecolor="white"
    )
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Mean net flow (arrivals - departures)")
    axes[1, 1].set_ylabel("Number of stations")
    axes[1, 1].set_title("Morning net flow distribution")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_demand_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '01_demand_overview.png'}")


def plot_station_distributions(station_stats: pd.DataFrame, flows: pd.DataFrame) -> None:
    top_stations = station_stats.nlargest(6, "mean_departures")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Morning Departure Distributions — Top 6 Stations", fontsize=13, fontweight="bold")

    for idx, (_, row) in enumerate(top_stations.iterrows()):
        ax = axes[idx // 3, idx % 3]
        sid = row["station_id"]

        station_morning = flows[
            (flows["station_id"] == sid)
            & (flows["hour"] >= MORNING_PEAK["start_hour"])
            & (flows["hour"] < MORNING_PEAK["end_hour"])
        ]
        daily_deps = station_morning.groupby("date")["departures"].sum()

        ax.hist(daily_deps, bins=20, density=True, alpha=0.6, color="#378ADD", edgecolor="white")

        mu = daily_deps.mean()
        sigma = daily_deps.std()
        x = np.linspace(daily_deps.min(), daily_deps.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label=f"N({mu:.1f}, {sigma:.1f})")

        lam = mu
        x_pois = np.arange(0, int(daily_deps.max()) + 5)
        ax.plot(x_pois, stats.poisson.pmf(x_pois, lam), "g--", linewidth=1.5, label=f"Pois({lam:.1f})")

        ax.set_title(row["name"][:30], fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel("Morning departures")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_station_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '02_station_distributions.png'}")


def main():
    stations = load_study_stations()
    trips = load_trips()
    trips = filter_study_area_trips(trips, stations)

    flows = compute_hourly_flows(trips, stations)
    flows.to_parquet(DATA_PROCESSED / "hourly_flows.parquet", index=False)

    station_stats = fit_demand_distributions(flows, stations)
    station_stats.to_parquet(DATA_PROCESSED / "station_demand_stats.parquet", index=False)
    station_stats.to_csv(DATA_PROCESSED / "station_demand_stats.csv", index=False)

    plot_system_overview(flows, station_stats)
    plot_station_distributions(station_stats, flows)

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE — Demand Profile Summary")
    print("=" * 60)
    print(f"Study area: {STUDY_AREA['name']}")
    print(f"Stations: {len(station_stats)}")
    print(f"Trip months: {', '.join(TRIP_MONTHS)}")
    print(f"Total study-area trips: {len(trips):,}")
    print(f"Net suppliers: {(station_stats['station_type'] == 'net_supplier').sum()}")
    print(f"Net receivers: {(station_stats['station_type'] == 'net_receiver').sum()}")
    print(f"Balanced: {(station_stats['station_type'] == 'balanced').sum()}")
    print(f"\nOutputs:")
    print(f"  {DATA_PROCESSED / 'hourly_flows.parquet'}")
    print(f"  {DATA_PROCESSED / 'station_demand_stats.parquet'}")
    print(f"  {FIGURES_DIR / '01_demand_overview.png'}")
    print(f"  {FIGURES_DIR / '02_station_distributions.png'}")


if __name__ == "__main__":
    main()
