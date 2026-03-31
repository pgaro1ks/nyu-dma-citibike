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
    COST_MODEL,
    OPPORTUNITY_COST,
    SIMULATION,
    TARGET_FILL_RATIO,
)

sns.set_theme(style="whitegrid", palette="muted")


def load_data():
    stations = pd.read_parquet(DATA_PROCESSED / "station_demand_stats.parquet")
    targets = pd.read_parquet(DATA_PROCESSED / "rebalancing_targets.parquet")
    opt_sweep = pd.read_parquet(DATA_PROCESSED / "optimization_sweep.parquet")
    return stations, targets, opt_sweep


def simulate_morning(
    stations,
    targets,
    rebalanced_inventory,
    rng,
    weather_factor=1.0,
    weekday=True,
):
    n_stations = len(stations)

    day_factor = 1.0 if weekday else 0.7
    demand_multiplier = weather_factor * day_factor

    departures = np.zeros(n_stations)
    arrivals = np.zeros(n_stations)

    for i in range(n_stations):
        dep_rate = max(stations.iloc[i]["mean_departures"] * demand_multiplier, 0.1)
        arr_rate = max(stations.iloc[i]["mean_arrivals"] * demand_multiplier, 0.1)

        if dep_rate > 20:
            departures[i] = max(0, rng.normal(dep_rate, stations.iloc[i].get("std_departures", dep_rate * 0.3)))
        else:
            departures[i] = rng.poisson(dep_rate)

        if arr_rate > 20:
            arrivals[i] = max(0, rng.normal(arr_rate, stations.iloc[i].get("std_arrivals", arr_rate * 0.3)))
        else:
            arrivals[i] = rng.poisson(arr_rate)

    bikes_available = rebalanced_inventory.copy()
    capacity = targets["capacity"].values.astype(float)

    unmet_departures = np.maximum(departures - bikes_available, 0)
    successful_departures = departures - unmet_departures
    bikes_after_dep = bikes_available - successful_departures

    docks_available = capacity - bikes_after_dep
    unmet_arrivals = np.maximum(arrivals - docks_available, 0)
    successful_arrivals = arrivals - unmet_arrivals

    total_unmet = unmet_departures.sum() + unmet_arrivals.sum()
    total_trips = departures.sum() + arrivals.sum()
    service_level = 1.0 - (total_unmet / max(total_trips, 1))

    return {
        "total_departures": departures.sum(),
        "total_arrivals": arrivals.sum(),
        "unmet_departures": unmet_departures.sum(),
        "unmet_arrivals": unmet_arrivals.sum(),
        "total_unmet": total_unmet,
        "service_level": service_level,
        "stations_with_unmet": (unmet_departures > 0).sum() + (unmet_arrivals > 0).sum(),
    }


def run_simulation(stations, targets, n_trucks, n_trials=5000, seed=42):
    rng = np.random.default_rng(seed)

    truck_capacity = n_trucks * COST_MODEL["truck_capacity_bikes"]

    base_inventory = targets["estimated_midnight_fill"].values.astype(float)
    target_inventory = targets["target_bikes"].values.astype(float)
    capacity = targets["capacity"].values.astype(float)

    deficit = target_inventory - base_inventory
    total_deficit_positive = np.maximum(deficit, 0).sum()
    fulfillment_ratio = min(1.0, truck_capacity / max(total_deficit_positive, 1))

    rebalanced = base_inventory + deficit * fulfillment_ratio
    rebalanced = np.clip(rebalanced, 0, capacity)

    weather_factors = rng.choice(
        [1.0, 0.85, 0.6, 1.1],
        size=n_trials,
        p=[0.65, 0.15, 0.10, 0.10],
    )
    weekday_flags = rng.choice([True, False], size=n_trials, p=[5 / 7, 2 / 7])

    results = []
    for trial in range(n_trials):
        morning = simulate_morning(
            stations, targets, rebalanced, rng,
            weather_factor=weather_factors[trial],
            weekday=weekday_flags[trial],
        )

        opp_cost = (
            morning["total_unmet"]
            * (OPPORTUNITY_COST["lost_trip_revenue_usd"] + OPPORTUNITY_COST["customer_dissatisfaction_penalty_usd"])
            * OPPORTUNITY_COST["brand_damage_multiplier"]
        )

        morning["opportunity_cost"] = opp_cost
        morning["weather_factor"] = weather_factors[trial]
        morning["is_weekday"] = weekday_flags[trial]
        morning["trial"] = trial
        results.append(morning)

    return pd.DataFrame(results)


def compute_summary_stats(sim_results, rebal_cost):
    sl = sim_results["service_level"]
    unmet = sim_results["total_unmet"]
    opp = sim_results["opportunity_cost"]
    net_benefit = opp - rebal_cost

    return {
        "mean_service_level": sl.mean(),
        "p10_service_level": sl.quantile(0.10),
        "p25_service_level": sl.quantile(0.25),
        "p50_service_level": sl.quantile(0.50),
        "p75_service_level": sl.quantile(0.75),
        "p90_service_level": sl.quantile(0.90),
        "mean_unmet_trips": unmet.mean(),
        "p90_unmet_trips": unmet.quantile(0.90),
        "mean_opportunity_cost": opp.mean(),
        "rebalancing_cost": rebal_cost,
        "mean_net_benefit": net_benefit.mean(),
        "p10_net_benefit": net_benefit.quantile(0.10),
        "p50_net_benefit": net_benefit.quantile(0.50),
        "p90_net_benefit": net_benefit.quantile(0.90),
        "prob_positive_net_benefit": (net_benefit > 0).mean(),
    }


def plot_simulation_results(sim_results, summary, n_trucks, rebal_cost):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Monte Carlo Simulation — {n_trucks} Trucks, {len(sim_results):,} Trials",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].hist(
        sim_results["service_level"], bins=50, color="#378ADD", alpha=0.7, edgecolor="white"
    )
    axes[0, 0].axvline(summary["mean_service_level"], color="red", linestyle="--", linewidth=2, label=f"Mean: {summary['mean_service_level']:.1%}")
    axes[0, 0].axvline(summary["p10_service_level"], color="orange", linestyle=":", linewidth=1.5, label=f"P10: {summary['p10_service_level']:.1%}")
    axes[0, 0].set_xlabel("Service Level")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Service Level Distribution")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].hist(
        sim_results["total_unmet"], bins=50, color="#E24B4A", alpha=0.7, edgecolor="white"
    )
    axes[0, 1].axvline(summary["mean_unmet_trips"], color="red", linestyle="--", linewidth=2, label=f"Mean: {summary['mean_unmet_trips']:.0f}")
    axes[0, 1].set_xlabel("Total Unmet Trips")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Unmet Demand Distribution")
    axes[0, 1].legend(fontsize=9)

    net_benefit = sim_results["opportunity_cost"] - rebal_cost
    axes[1, 0].hist(
        net_benefit, bins=50, color="#639922", alpha=0.7, edgecolor="white"
    )
    axes[1, 0].axvline(0, color="black", linestyle="-", linewidth=1)
    axes[1, 0].axvline(summary["mean_net_benefit"], color="red", linestyle="--", linewidth=2, label=f"Mean: ${summary['mean_net_benefit']:,.0f}")
    axes[1, 0].set_xlabel("Net Benefit ($)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Net Benefit (Opportunity Cost Saved − Rebalancing Cost)")
    axes[1, 0].legend(fontsize=9)

    percentiles = [10, 25, 50, 75, 90]
    sl_vals = [sim_results["service_level"].quantile(p / 100) for p in percentiles]
    unmet_vals = [sim_results["total_unmet"].quantile(p / 100) for p in percentiles]
    nb_vals = [net_benefit.quantile(p / 100) for p in percentiles]

    cell_text = []
    for i, p in enumerate(percentiles):
        cell_text.append([
            f"P{p}",
            f"{sl_vals[i]:.1%}",
            f"{unmet_vals[i]:.0f}",
            f"${nb_vals[i]:,.0f}",
        ])

    axes[1, 1].axis("off")
    table = axes[1, 1].table(
        cellText=cell_text,
        colLabels=["Percentile", "Service Level", "Unmet Trips", "Net Benefit"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    axes[1, 1].set_title("Percentile Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_simulation_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '05_simulation_results.png'}")


def main():
    stations, targets, opt_sweep = load_data()

    if opt_sweep[opt_sweep["service_level"] >= 0.85].shape[0] > 0:
        viable = opt_sweep[opt_sweep["service_level"] >= 0.85]
        optimal_row = viable.loc[viable["total_cost"].idxmin()]
        n_trucks = int(optimal_row["n_trucks"])
    else:
        n_trucks = 4
    optimal_row = opt_sweep[opt_sweep["n_trucks"] == n_trucks].iloc[0]
    rebal_cost = optimal_row["total_cost"]

    print(f"[simulate] Running {SIMULATION['n_trials']:,} trials with {n_trucks} trucks")
    print(f"  Rebalancing cost: ${rebal_cost:,.2f}")

    sim_results = run_simulation(
        stations, targets, n_trucks,
        n_trials=SIMULATION["n_trials"],
        seed=SIMULATION["random_seed"],
    )

    summary = compute_summary_stats(sim_results, rebal_cost)

    sim_results.to_parquet(DATA_PROCESSED / "simulation_results.parquet", index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_parquet(DATA_PROCESSED / "simulation_summary.parquet", index=False)
    summary_df.to_csv(DATA_PROCESSED / "simulation_summary.csv", index=False)

    plot_simulation_results(sim_results, summary, n_trucks, rebal_cost)

    print(f"\n{'=' * 60}")
    print(f"PHASE 4 COMPLETE — Monte Carlo Simulation")
    print(f"{'=' * 60}")
    print(f"Trucks: {n_trucks}")
    print(f"Trials: {SIMULATION['n_trials']:,}")
    print(f"Mean service level:   {summary['mean_service_level']:.1%}")
    print(f"P10 service level:    {summary['p10_service_level']:.1%}")
    print(f"Mean unmet trips:     {summary['mean_unmet_trips']:.0f}")
    print(f"Rebalancing cost:     ${rebal_cost:,.2f}")
    print(f"Mean opp cost saved:  ${summary['mean_opportunity_cost']:,.2f}")
    print(f"Mean net benefit:     ${summary['mean_net_benefit']:,.2f}")
    print(f"P(net benefit > 0):   {summary['prob_positive_net_benefit']:.1%}")


if __name__ == "__main__":
    main()
