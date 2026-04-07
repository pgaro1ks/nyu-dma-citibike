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
)

sns.set_theme(style="whitegrid", palette="muted")


def load_data():
    stations = pd.read_parquet(DATA_PROCESSED / "station_demand_stats.parquet")
    targets = pd.read_parquet(DATA_PROCESSED / "rebalancing_targets.parquet")
    opt_sweep = pd.read_parquet(DATA_PROCESSED / "optimization_sweep.parquet")
    return stations, targets, opt_sweep


def quick_simulate(stations, targets, n_trucks, n_trials=2000, seed=42):
    from src_06_sim_helper import run_simulation_quick
    return run_simulation_quick(stations, targets, n_trucks, n_trials, seed)


def run_truck_sweep(stations, targets, opt_sweep):
    rng = np.random.default_rng(SIMULATION["random_seed"])

    truck_capacity = COST_MODEL["truck_capacity_bikes"]
    base_inventory = targets["estimated_midnight_fill"].values.astype(float)
    target_inventory = targets["target_bikes"].values.astype(float)
    capacity = targets["capacity"].values.astype(float)
    deficit = target_inventory - base_inventory
    total_deficit_positive = np.maximum(deficit, 0).sum()

    results = []
    for n_trucks in range(1, 9):
        opt_row = opt_sweep[opt_sweep["n_trucks"] == n_trucks]
        if len(opt_row) == 0:
            continue
        rebal_cost = float(opt_row["total_cost"].iloc[0])
        rebal_service = float(opt_row["service_level"].iloc[0])

        bikes_available = n_trucks * truck_capacity
        fulfillment = min(1.0, bikes_available / max(total_deficit_positive, 1))

        rebalanced = base_inventory + deficit * fulfillment
        rebalanced = np.clip(rebalanced, 0, capacity)

        trial_results = []
        for trial in range(2000):
            weather = rng.choice([1.0, 0.85, 0.6, 1.1], p=[0.65, 0.15, 0.10, 0.10])
            weekday = rng.random() < 5 / 7

            demand_mult = weather * (1.0 if weekday else 0.7)

            departures = np.array([
                rng.poisson(max(0.1, stations.iloc[i]["mean_departures"] * demand_mult))
                for i in range(len(stations))
            ])
            arrivals = np.array([
                rng.poisson(max(0.1, stations.iloc[i]["mean_arrivals"] * demand_mult))
                for i in range(len(stations))
            ])

            unmet_dep = np.maximum(departures - rebalanced, 0).sum()
            docks = capacity - (rebalanced - np.minimum(departures, rebalanced))
            unmet_arr = np.maximum(arrivals - docks, 0).sum()
            total_unmet = unmet_dep + unmet_arr
            total_trips = departures.sum() + arrivals.sum()
            sl = 1.0 - total_unmet / max(total_trips, 1)

            opp = total_unmet * (
                OPPORTUNITY_COST["lost_trip_revenue_usd"]
                + OPPORTUNITY_COST["customer_dissatisfaction_penalty_usd"]
            ) * OPPORTUNITY_COST["brand_damage_multiplier"]

            trial_results.append({
                "service_level": sl,
                "unmet": total_unmet,
                "opp_cost": opp,
                "net_benefit": opp - rebal_cost,
            })

        trial_df = pd.DataFrame(trial_results)
        results.append({
            "n_trucks": n_trucks,
            "rebal_cost": rebal_cost,
            "mean_service_level": trial_df["service_level"].mean(),
            "p10_service_level": trial_df["service_level"].quantile(0.10),
            "p90_service_level": trial_df["service_level"].quantile(0.90),
            "mean_unmet": trial_df["unmet"].mean(),
            "mean_opp_cost": trial_df["opp_cost"].mean(),
            "mean_net_benefit": trial_df["net_benefit"].mean(),
            "p10_net_benefit": trial_df["net_benefit"].quantile(0.10),
            "prob_positive": (trial_df["net_benefit"] > 0).mean(),
        })

    return pd.DataFrame(results)


def run_parameter_sensitivity(stations, targets, opt_sweep):
    rng = np.random.default_rng(SIMULATION["random_seed"])

    base_inventory = targets["estimated_midnight_fill"].values.astype(float)
    target_inventory = targets["target_bikes"].values.astype(float)
    capacity = targets["capacity"].values.astype(float)
    deficit = target_inventory - base_inventory
    total_deficit_positive = np.maximum(deficit, 0).sum()

    base_n_trucks = 4
    base_cost = float(opt_sweep[opt_sweep["n_trucks"] == base_n_trucks]["total_cost"].iloc[0])
    base_miles = float(opt_sweep[opt_sweep["n_trucks"] == base_n_trucks]["total_miles"].iloc[0])

    sensitivities = {}

    cap_values = [30, 40, 48, 55, 60]
    cap_results = []
    for cap in cap_values:
        bikes_avail = base_n_trucks * cap
        fulfillment = min(1.0, bikes_avail / max(total_deficit_positive, 1))

        adjusted_cost = (
            base_n_trucks * COST_MODEL["fixed_cost_per_truck_per_night_usd"]
            + base_miles * COST_MODEL["variable_cost_per_mile_usd"]
            + base_miles / COST_MODEL["truck_mpg"] * COST_MODEL["fuel_cost_per_gallon_usd"]
            + 3.0 * COST_MODEL["driver_hourly_wage_usd"] * base_n_trucks
        )

        rebalanced = base_inventory + deficit * fulfillment
        rebalanced = np.clip(rebalanced, 0, capacity)

        sls = []
        for _ in range(500):
            demand_mult = rng.choice([1.0, 0.85, 0.6, 1.1], p=[0.65, 0.15, 0.10, 0.10])
            deps = np.array([rng.poisson(max(0.1, stations.iloc[i]["mean_departures"] * demand_mult)) for i in range(len(stations))])
            arrs = np.array([rng.poisson(max(0.1, stations.iloc[i]["mean_arrivals"] * demand_mult)) for i in range(len(stations))])
            unmet = np.maximum(deps - rebalanced, 0).sum() + np.maximum(arrs - (capacity - rebalanced + np.minimum(deps, rebalanced)), 0).sum()
            total = deps.sum() + arrs.sum()
            sls.append(1.0 - unmet / max(total, 1))

        cap_results.append({"truck_capacity": cap, "cost": adjusted_cost, "service_level": np.mean(sls)})
    sensitivities["truck_capacity"] = pd.DataFrame(cap_results)

    wage_values = [60, 70, 82, 90, 100]  # 2-worker crew loaded hourly rate
    wage_results = []
    for wage in wage_values:
        adjusted_cost = (
            base_n_trucks * COST_MODEL["fixed_cost_per_truck_per_night_usd"]
            + base_miles * COST_MODEL["variable_cost_per_mile_usd"]
            + base_miles / COST_MODEL["truck_mpg"] * COST_MODEL["fuel_cost_per_gallon_usd"]
            + 3.0 * wage * base_n_trucks
        )
        wage_results.append({"driver_wage": wage, "cost": adjusted_cost})
    sensitivities["driver_wage"] = pd.DataFrame(wage_results)

    fuel_values = [3.00, 3.50, 4.00, 4.50]
    fuel_results = []
    for fuel in fuel_values:
        adjusted_cost = (
            base_n_trucks * COST_MODEL["fixed_cost_per_truck_per_night_usd"]
            + base_miles * COST_MODEL["variable_cost_per_mile_usd"]
            + base_miles / COST_MODEL["truck_mpg"] * fuel
            + 3.0 * COST_MODEL["driver_hourly_wage_usd"] * base_n_trucks
        )
        fuel_results.append({"fuel_cost": fuel, "cost": adjusted_cost})
    sensitivities["fuel_cost"] = pd.DataFrame(fuel_results)

    return sensitivities


def plot_efficient_frontier(truck_sweep):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle("Efficient Frontier — Trucks vs. Service Level vs. Cost", fontsize=14, fontweight="bold")

    color1 = "#378ADD"
    color2 = "#E24B4A"
    color3 = "#639922"

    ax1.set_xlabel("Number of Trucks", fontsize=12)
    ax1.set_ylabel("Service Level", fontsize=12, color=color1)
    ax1.plot(
        truck_sweep["n_trucks"], truck_sweep["mean_service_level"],
        "o-", color=color1, linewidth=2.5, markersize=10, label="Mean service level"
    )
    ax1.fill_between(
        truck_sweep["n_trucks"],
        truck_sweep["p10_service_level"],
        truck_sweep["p90_service_level"],
        alpha=0.15, color=color1, label="P10–P90 range"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.5, 1.05)
    ax1.axhline(0.90, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.text(0.8, 0.905, "90% target", fontsize=9, color="gray")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Cost per Night ($)", fontsize=12, color=color2)
    ax2.plot(
        truck_sweep["n_trucks"], truck_sweep["rebal_cost"],
        "s--", color=color2, linewidth=2, markersize=8, label="Rebalancing cost"
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    for _, row in truck_sweep.iterrows():
        ax1.annotate(
            f"${row['mean_net_benefit']:,.0f}\nnet/night",
            xy=(row["n_trucks"], row["mean_service_level"]),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=8,
            color=color3,
            fontweight="bold",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    ax1.set_xticks(truck_sweep["n_trucks"].values)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '06_efficient_frontier.png'}")


def plot_tornado(sensitivities, base_cost):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Sensitivity Tornado — Impact on Total Nightly Cost", fontsize=14, fontweight="bold")

    bars = []

    cap_df = sensitivities["truck_capacity"]
    low_cap = cap_df["cost"].min()
    high_cap = cap_df["cost"].max()
    bars.append(("Truck Capacity\n(30–60 bikes)", low_cap, high_cap))

    wage_df = sensitivities["driver_wage"]
    low_wage = wage_df["cost"].min()
    high_wage = wage_df["cost"].max()
    bars.append(("Crew Wage\n($60–$100/hr)", low_wage, high_wage))

    fuel_df = sensitivities["fuel_cost"]
    low_fuel = fuel_df["cost"].min()
    high_fuel = fuel_df["cost"].max()
    bars.append(("Fuel Cost\n($3.00–$4.50/gal)", low_fuel, high_fuel))

    bars.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)

    y_pos = range(len(bars))
    for i, (label, low, high) in enumerate(bars):
        ax.barh(i, high - base_cost, left=base_cost, height=0.5, color="#E24B4A", alpha=0.7)
        ax.barh(i, base_cost - low, left=low, height=0.5, color="#378ADD", alpha=0.7)
        ax.text(high + 20, i, f"${high:,.0f}", va="center", fontsize=9, color="#E24B4A")
        ax.text(low - 20, i, f"${low:,.0f}", va="center", fontsize=9, color="#378ADD", ha="right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([b[0] for b in bars])
    ax.axvline(base_cost, color="black", linestyle="-", linewidth=1.5)
    ax.set_xlabel("Total Nightly Cost ($)")
    ax.set_title(f"Base cost: ${base_cost:,.0f} (4 trucks)")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_sensitivity_tornado.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '07_sensitivity_tornado.png'}")


def main():
    stations, targets, opt_sweep = load_data()

    print("[sensitivity] Running truck count sweep with simulation...")
    truck_sweep = run_truck_sweep(stations, targets, opt_sweep)
    truck_sweep.to_parquet(DATA_PROCESSED / "sensitivity_truck_sweep.parquet", index=False)
    truck_sweep.to_csv(DATA_PROCESSED / "sensitivity_truck_sweep.csv", index=False)

    print("[sensitivity] Running parameter sensitivity analysis...")
    sensitivities = run_parameter_sensitivity(stations, targets, opt_sweep)
    for key, df in sensitivities.items():
        df.to_csv(DATA_PROCESSED / f"sensitivity_{key}.csv", index=False)

    plot_efficient_frontier(truck_sweep)

    base_cost = float(opt_sweep[opt_sweep["n_trucks"] == 4]["total_cost"].iloc[0])
    plot_tornado(sensitivities, base_cost)

    print(f"\n{'=' * 60}")
    print("PHASE 5 COMPLETE — Sensitivity & Efficient Frontier")
    print("=" * 60)
    print("\nTruck sweep results:")
    for _, row in truck_sweep.iterrows():
        print(
            f"  {int(row['n_trucks'])} trucks: "
            f"SL={row['mean_service_level']:.1%}, "
            f"Cost=${row['rebal_cost']:,.0f}, "
            f"Net=${row['mean_net_benefit']:,.0f}/night, "
            f"P(>0)={row['prob_positive']:.0%}"
        )

    best = truck_sweep.loc[truck_sweep["mean_net_benefit"].idxmax()]
    print(f"\n[recommendation] {int(best['n_trucks'])} trucks maximizes net benefit")
    print(f"  Service level: {best['mean_service_level']:.1%}")
    print(f"  Net benefit: ${best['mean_net_benefit']:,.0f}/night")
    print(f"  Monthly savings: ${best['mean_net_benefit'] * 30:,.0f}")


if __name__ == "__main__":
    main()
