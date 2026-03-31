import sys
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    TARGET_FILL_RATIO,
    COST_MODEL,
    OPPORTUNITY_COST,
)

sns.set_theme(style="whitegrid", palette="muted")


def load_data():
    flows = pd.read_parquet(DATA_PROCESSED / "hourly_flows.parquet")
    stations = pd.read_parquet(DATA_PROCESSED / "station_demand_stats.parquet")
    return flows, stations


def compute_midnight_inventory(flows: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    daily_net = (
        flows.groupby(["station_id", "date"])["net_flow"]
        .sum()
        .reset_index()
    )

    avg_daily_net = (
        daily_net.groupby("station_id")["net_flow"]
        .mean()
        .reset_index()
        .rename(columns={"net_flow": "avg_daily_net_flow"})
    )

    result = stations[["station_id", "name", "capacity", "lat", "lon", "station_type"]].merge(
        avg_daily_net, on="station_id", how="left"
    )

    result["estimated_midnight_fill"] = np.clip(
        result["capacity"] * 0.5 + result["avg_daily_net_flow"],
        0,
        result["capacity"],
    ).round().astype(int)

    result["midnight_fill_ratio"] = result["estimated_midnight_fill"] / result["capacity"]

    return result


def compute_morning_target(midnight: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    target = midnight.merge(
        stations[["station_id", "mean_departures", "mean_arrivals", "mean_net"]],
        on="station_id",
        how="left",
    )

    ideal = TARGET_FILL_RATIO["ideal"]
    target["target_bikes"] = (target["capacity"] * ideal).round().astype(int)

    target["target_bikes"] = np.where(
        target["mean_net"] < -2,
        np.clip(
            (target["capacity"] * 0.65).round(),
            target["mean_departures"].round(),
            target["capacity"],
        ),
        target["target_bikes"],
    ).astype(int)

    target["target_bikes"] = np.where(
        target["mean_net"] > 2,
        np.clip(
            (target["capacity"] * 0.35).round(),
            0,
            target["capacity"] - target["mean_arrivals"].round(),
        ),
        target["target_bikes"],
    ).astype(int)

    target["deficit"] = target["target_bikes"] - target["estimated_midnight_fill"]
    target["needs_pickup"] = np.where(target["deficit"] < 0, -target["deficit"], 0).astype(int)
    target["needs_dropoff"] = np.where(target["deficit"] > 0, target["deficit"], 0).astype(int)

    return target


def estimate_no_rebalancing_cost(target: pd.DataFrame) -> dict:
    unmet_departures = np.maximum(
        target["mean_departures"] - target["estimated_midnight_fill"], 0
    ).sum()

    unmet_arrivals = np.maximum(
        target["mean_arrivals"] - (target["capacity"] - target["estimated_midnight_fill"]), 0
    ).sum()

    lost_revenue = (unmet_departures + unmet_arrivals) * OPPORTUNITY_COST["lost_trip_revenue_usd"]
    dissatisfaction = (unmet_departures + unmet_arrivals) * OPPORTUNITY_COST["customer_dissatisfaction_penalty_usd"]
    total = (lost_revenue + dissatisfaction) * OPPORTUNITY_COST["brand_damage_multiplier"]

    return {
        "unmet_departures_per_morning": unmet_departures,
        "unmet_arrivals_per_morning": unmet_arrivals,
        "daily_lost_revenue": lost_revenue,
        "daily_dissatisfaction_cost": dissatisfaction,
        "daily_total_opportunity_cost": total,
        "monthly_opportunity_cost": total * 30,
    }


def plot_imbalance(target: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Station Imbalance Analysis — Midnight vs. Morning Target", fontsize=14, fontweight="bold")

    scatter = axes[0].scatter(
        target["lon"], target["lat"],
        c=target["deficit"],
        cmap="RdBu",
        s=target["capacity"] * 3,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
        vmin=-target["deficit"].abs().max(),
        vmax=target["deficit"].abs().max(),
    )
    plt.colorbar(scatter, ax=axes[0], label="Deficit (+ needs bikes, - has surplus)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Rebalancing needs by location")

    sorted_t = target.sort_values("deficit")
    colors = ["#E24B4A" if d < 0 else "#378ADD" for d in sorted_t["deficit"]]
    axes[1].barh(sorted_t["name"].str[:25], sorted_t["deficit"], color=colors, alpha=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Deficit (bikes needed)")
    axes[1].set_title("Per-station rebalancing demand")
    axes[1].tick_params(axis="y", labelsize=6)

    total_pickup = target["needs_pickup"].sum()
    total_dropoff = target["needs_dropoff"].sum()
    truck_cap = COST_MODEL["truck_capacity_bikes"]
    min_truckloads = max(total_pickup, total_dropoff) / truck_cap

    labels = ["Bikes to pick up", "Bikes to drop off", f"Min truckloads\n(cap={truck_cap})"]
    values = [total_pickup, total_dropoff, min_truckloads]
    bar_colors = ["#E24B4A", "#378ADD", "#639922"]
    axes[2].bar(labels, values, color=bar_colors, alpha=0.8, edgecolor="white")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Total rebalancing volume")
    for i, v in enumerate(values):
        axes[2].text(i, v + 1, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_imbalance_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '03_imbalance_analysis.png'}")


def main():
    flows, stations = load_data()

    midnight = compute_midnight_inventory(flows, stations)
    target = compute_morning_target(midnight, stations)

    target.to_parquet(DATA_PROCESSED / "rebalancing_targets.parquet", index=False)
    target.to_csv(DATA_PROCESSED / "rebalancing_targets.csv", index=False)

    no_rebal = estimate_no_rebalancing_cost(target)

    plot_imbalance(target)

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE — Imbalance Analysis")
    print("=" * 60)
    print(f"Stations analyzed: {len(target)}")
    print(f"Stations needing pickup:  {(target['needs_pickup'] > 0).sum()}")
    print(f"Stations needing dropoff: {(target['needs_dropoff'] > 0).sum()}")
    print(f"Total bikes to move:      {target['needs_pickup'].sum() + target['needs_dropoff'].sum()}")
    print(f"\nCost of doing nothing (per morning):")
    print(f"  Unmet departures: {no_rebal['unmet_departures_per_morning']:.0f} trips")
    print(f"  Unmet arrivals:   {no_rebal['unmet_arrivals_per_morning']:.0f} trips")
    print(f"  Lost revenue:     ${no_rebal['daily_lost_revenue']:.2f}")
    print(f"  Dissatisfaction:  ${no_rebal['daily_dissatisfaction_cost']:.2f}")
    print(f"  TOTAL daily cost: ${no_rebal['daily_total_opportunity_cost']:.2f}")
    print(f"  Monthly cost:     ${no_rebal['monthly_opportunity_cost']:,.2f}")


if __name__ == "__main__":
    main()
