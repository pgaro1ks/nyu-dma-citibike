import sys
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    COST_MODEL,
    MAX_TRUCKS,
    MAX_STATIONS,
)


def load_data():
    targets = pd.read_parquet(DATA_PROCESSED / "rebalancing_targets.parquet")
    return targets


def manhattan_distance_miles(lat1, lon1, lat2, lon2):
    lat_diff = abs(lat2 - lat1) * 69.0
    lon_diff = abs(lon2 - lon1) * 52.3
    return lat_diff + lon_diff


def build_distance_matrix(stations):
    coords = stations[["lat", "lon"]].values
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = manhattan_distance_miles(
                coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
            )
    return dist


def nearest_neighbor_tsp(dist_matrix, station_indices, start_idx=None):
    if len(station_indices) <= 1:
        return list(station_indices), 0.0

    remaining = set(station_indices)
    if start_idx is None:
        start_idx = station_indices[0]
    route = [start_idx]
    remaining.discard(start_idx)
    total_dist = 0.0

    while remaining:
        current = route[-1]
        nearest = min(remaining, key=lambda x: dist_matrix[current, x])
        total_dist += dist_matrix[current, nearest]
        route.append(nearest)
        remaining.discard(nearest)

    return route, total_dist


def two_opt_improve(route, dist_matrix, max_iterations=100):
    if len(route) <= 3:
        return route, sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(route) - 2):
            for j in range(i + 2, len(route)):
                d1 = dist_matrix[route[i], route[i + 1]] + dist_matrix[route[j - 1], route[j] if j < len(route) else route[0]]
                d2 = dist_matrix[route[i], route[j - 1]] + dist_matrix[route[i + 1], route[j] if j < len(route) else route[0]]
                if d2 < d1 - 1e-10:
                    route[i + 1:j] = route[i + 1:j][::-1]
                    improved = True

    total = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    return route, total


def assign_stations_to_trucks(targets, n_trucks, dist_matrix):
    pickup_stations = targets[targets["needs_pickup"] > 0].index.tolist()
    dropoff_stations = targets[targets["needs_dropoff"] > 0].index.tolist()

    all_active = list(set(pickup_stations + dropoff_stations))
    if not all_active:
        return {k: [] for k in range(n_trucks)}

    coords = targets.loc[all_active, ["lat", "lon"]].values
    n = len(all_active)

    assignments = {k: [] for k in range(n_trucks)}

    sorted_by_lon = sorted(all_active, key=lambda i: targets.loc[i, "lon"])

    chunk_size = max(1, len(sorted_by_lon) // n_trucks)
    for k in range(n_trucks):
        start = k * chunk_size
        if k == n_trucks - 1:
            assignments[k] = sorted_by_lon[start:]
        else:
            assignments[k] = sorted_by_lon[start:start + chunk_size]

    return assignments


def simulate_truck_route(route_indices, targets, dist_matrix):
    if not route_indices:
        return {"route": [], "miles": 0, "bikes_moved": 0, "stops": 0, "time_hours": 0}

    pickup_indices = [i for i in route_indices if targets.loc[i, "needs_pickup"] > 0]
    dropoff_indices = [i for i in route_indices if targets.loc[i, "needs_dropoff"] > 0]

    ordered = pickup_indices + dropoff_indices

    if len(ordered) <= 1:
        miles = 0
    else:
        ordered, miles = nearest_neighbor_tsp(dist_matrix, ordered)
        ordered, miles = two_opt_improve(ordered, dist_matrix)

    bikes_picked = sum(targets.loc[i, "needs_pickup"] for i in ordered if targets.loc[i, "needs_pickup"] > 0)
    bikes_dropped = sum(targets.loc[i, "needs_dropoff"] for i in ordered if targets.loc[i, "needs_dropoff"] > 0)

    n_stops = len(ordered)
    stop_time_hours = n_stops * COST_MODEL["avg_load_unload_minutes_per_stop"] / 60
    drive_time_hours = miles / COST_MODEL["avg_speed_mph_overnight"] if miles > 0 else 0
    total_hours = stop_time_hours + drive_time_hours

    return {
        "route": ordered,
        "miles": miles,
        "bikes_picked_up": bikes_picked,
        "bikes_dropped_off": bikes_dropped,
        "bikes_moved": bikes_picked + bikes_dropped,
        "stops": n_stops,
        "drive_hours": drive_time_hours,
        "stop_hours": stop_time_hours,
        "total_hours": total_hours,
    }


def compute_route_cost(route_result):
    fixed = COST_MODEL["fixed_cost_per_truck_per_night_usd"]
    miles_cost = route_result["miles"] * COST_MODEL["variable_cost_per_mile_usd"]
    fuel_cost = (route_result["miles"] / COST_MODEL["truck_mpg"]) * COST_MODEL["fuel_cost_per_gallon_usd"]

    hours = route_result["total_hours"]
    if hours <= COST_MODEL["overtime_threshold_hours"]:
        labor = hours * COST_MODEL["driver_hourly_wage_usd"]
    else:
        regular = COST_MODEL["overtime_threshold_hours"] * COST_MODEL["driver_hourly_wage_usd"]
        ot = (hours - COST_MODEL["overtime_threshold_hours"]) * COST_MODEL["driver_hourly_wage_usd"] * COST_MODEL["overtime_multiplier"]
        labor = regular + ot

    total = fixed + miles_cost + fuel_cost + labor

    return {
        "fixed_cost": fixed,
        "mileage_cost": miles_cost,
        "fuel_cost": fuel_cost,
        "labor_cost": labor,
        "total_cost": total,
    }


def optimize(targets, n_trucks, dist_matrix):
    assignments = assign_stations_to_trucks(targets, n_trucks, dist_matrix)

    truck_results = []
    total_cost = 0
    total_bikes = 0
    total_miles = 0

    for k in range(n_trucks):
        route_result = simulate_truck_route(assignments[k], targets, dist_matrix)
        cost_result = compute_route_cost(route_result)
        route_result.update(cost_result)
        route_result["truck_id"] = k + 1
        truck_results.append(route_result)
        total_cost += cost_result["total_cost"]
        total_bikes += route_result["bikes_moved"]
        total_miles += route_result["miles"]

    return {
        "n_trucks": n_trucks,
        "truck_results": truck_results,
        "total_cost": total_cost,
        "total_bikes_moved": total_bikes,
        "total_miles": total_miles,
    }


def compute_service_level(targets, optimization_result):
    total_deficit = targets["needs_dropoff"].sum()
    total_surplus = targets["needs_pickup"].sum()
    bikes_capacity = optimization_result["n_trucks"] * COST_MODEL["truck_capacity_bikes"]

    fulfillment = min(1.0, bikes_capacity / max(total_deficit, 1))

    all_visited = set()
    for r in optimization_result["truck_results"]:
        all_visited.update(r["route"])
    stations_needing_service = set(targets[targets["needs_pickup"] > 0].index) | set(targets[targets["needs_dropoff"] > 0].index)
    station_coverage = len(all_visited & stations_needing_service) / max(len(stations_needing_service), 1)

    service_level = min(1.0, (fulfillment * 0.5 + station_coverage * 0.5))
    return service_level


def plot_optimal_routes(targets, optimization_result):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(
        f"Optimal Truck Routes — {optimization_result['n_trucks']} Trucks, "
        f"${optimization_result['total_cost']:,.0f} Total Cost",
        fontsize=14,
        fontweight="bold",
    )

    colors_map = {"net_supplier": "#E24B4A", "net_receiver": "#378ADD", "balanced": "#888780"}
    for stype, color in colors_map.items():
        subset = targets[targets["station_type"] == stype]
        ax.scatter(
            subset["lon"], subset["lat"],
            s=subset["capacity"] * 4,
            c=color, alpha=0.4, label=stype, edgecolors="white", linewidth=0.5,
        )

    truck_colors = cm.Set1(np.linspace(0, 1, max(optimization_result["n_trucks"], 1)))
    for result in optimization_result["truck_results"]:
        if not result["route"]:
            continue
        route = result["route"]
        lats = [targets.loc[i, "lat"] for i in route]
        lons = [targets.loc[i, "lon"] for i in route]
        color = truck_colors[result["truck_id"] - 1]
        ax.plot(
            lons, lats, "-o",
            color=color, linewidth=2, markersize=6, alpha=0.8,
            label=f"Truck {result['truck_id']} ({result['miles']:.1f} mi, {result['stops']} stops)",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Station types + truck routes")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_optimal_routes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '04_optimal_routes.png'}")


def main():
    targets = load_data()

    dist_matrix = build_distance_matrix(targets)

    best_cost = float("inf")
    best_result = None
    best_n = 0

    print("[optimize] Sweeping truck counts 1 through 8...")
    results_log = []
    for n in range(1, 9):
        result = optimize(targets, n, dist_matrix)
        sl = compute_service_level(targets, result)
        result["service_level"] = sl
        results_log.append({
            "n_trucks": n,
            "total_cost": result["total_cost"],
            "total_miles": result["total_miles"],
            "total_bikes": result["total_bikes_moved"],
            "service_level": sl,
        })
        print(
            f"  {n} trucks: ${result['total_cost']:,.0f} cost, "
            f"{result['total_miles']:.1f} mi, "
            f"{result['total_bikes_moved']} bikes, "
            f"{sl:.1%} service"
        )

        if sl >= 0.85 and result["total_cost"] < best_cost:
            best_cost = result["total_cost"]
            best_result = result
            best_n = n

    if best_result is None:
        best_result = optimize(targets, 4, dist_matrix)
        best_result["service_level"] = compute_service_level(targets, best_result)
        best_n = 4

    print(f"\n[optimal] {best_n} trucks selected")
    print(f"  Total cost: ${best_result['total_cost']:,.2f}")
    print(f"  Service level: {best_result['service_level']:.1%}")

    for r in best_result["truck_results"]:
        route_names = [targets.loc[i, "name"][:25] for i in r["route"]] if r["route"] else []
        print(
            f"  Truck {r['truck_id']}: {r['stops']} stops, "
            f"{r['miles']:.1f} mi, ${r['total_cost']:,.0f}, "
            f"{r['bikes_moved']} bikes"
        )

    results_df = pd.DataFrame(results_log)
    results_df.to_parquet(DATA_PROCESSED / "optimization_sweep.parquet", index=False)
    results_df.to_csv(DATA_PROCESSED / "optimization_sweep.csv", index=False)

    truck_detail = pd.DataFrame(best_result["truck_results"])
    truck_detail.to_parquet(DATA_PROCESSED / "optimal_routes.parquet", index=False)
    truck_detail.to_csv(DATA_PROCESSED / "optimal_routes.csv", index=False)

    plot_optimal_routes(targets, best_result)

    print(f"\n[saved] optimization_sweep.parquet, optimal_routes.parquet")


if __name__ == "__main__":
    main()
