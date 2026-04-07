import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PROCESSED, FIGURES_DIR

sns.set_theme(style="whitegrid", palette="muted")


BIKE_SPECS = {
    "classic": {
        "length_in": 68,
        "width_in": 24,
        "height_in": 42,
        "weight_lbs": 45,
    },
    "ebike": {
        "length_in": 70,
        "width_in": 25,
        "height_in": 44,
        "weight_lbs": 65,
    },
    "fleet_ebike_share": 0.706,
    "fleet_classic_share": 0.294,
}

TRUCK_CANDIDATES = {
    "isuzu_npr_hd_16ft": {
        "name": "Isuzu NPR-HD 16' Box",
        "cargo_length_in": 192,
        "cargo_width_in": 92,
        "cargo_height_in": 91,
        "gvwr_lbs": 14500,
        "curb_weight_lbs": 7500,
        "payload_lbs": 7000,
        "msrp_usd": 65000,
        "mpg_city_loaded": 9.5,
        "fuel_tank_gal": 38.6,
        "engine": "GM 6.6L V8 Gas, 350 HP",
        "cdl_required": False,
    },
    "isuzu_npr_hd_18ft": {
        "name": "Isuzu NPR-HD 18' Box",
        "cargo_length_in": 216,
        "cargo_width_in": 92,
        "cargo_height_in": 91,
        "gvwr_lbs": 14500,
        "curb_weight_lbs": 7800,
        "payload_lbs": 6700,
        "msrp_usd": 73000,
        "mpg_city_loaded": 8.5,
        "fuel_tank_gal": 38.6,
        "engine": "GM 6.6L V8 Gas, 350 HP",
        "cdl_required": False,
    },
    "ford_e450_16ft": {
        "name": "Ford E-450 16' Box",
        "cargo_length_in": 192,
        "cargo_width_in": 90,
        "cargo_height_in": 85,
        "gvwr_lbs": 14500,
        "curb_weight_lbs": 7400,
        "payload_lbs": 7100,
        "msrp_usd": 62000,
        "mpg_city_loaded": 8.0,
        "fuel_tank_gal": 55,
        "engine": "Ford 7.3L V8 Gas, 350 HP",
        "cdl_required": False,
    },
}

RACK_CONFIG = {
    "method": "vertical_wheel_hang",
    "footprint_per_bike_length_in": 6,
    "footprint_per_bike_width_in": 24,
    "vertical_clearance_needed_in": 72,
    "row_spacing_in": 6,
    "wall_clearance_in": 3,
    "aisle_width_in": 24,
    "rack_weight_lbs": 200,
    "rack_install_cost_usd": 3500,
}

FUEL = {
    "nj_regular_per_gal_usd": 3.30,
    "ny_regular_per_gal_usd": 4.07,
    "refuel_location": "NJ",
}

LABOR = {
    "base_hourly_wage_usd": 28.00,
    "night_shift_differential_pct": 0.12,
    "payroll_tax_benefits_multiplier": 1.30,
    "shift_start_hour": 0,
    "shift_end_hour": 5,
    "shift_duration_hours": 5,
    "pre_post_shift_overhead_hours": 0.75,
    "workers_per_truck": 2,
}

LEASE = {
    "monthly_rate_16ft_usd": 1800,
    "monthly_rate_18ft_usd": 2100,
    "includes_maintenance": True,
    "lease_term_months": 36,
    "insurance_monthly_usd": 350,
}

BUY_NEW = {
    "down_payment_pct": 0.20,
    "loan_apr": 0.065,
    "loan_term_months": 60,
    "depreciation_years": 7,
    "salvage_value_pct": 0.15,
    "annual_maintenance_usd": 4500,
    "annual_insurance_usd": 4200,
}

BUY_USED = {
    "price_pct_of_msrp": 0.55,
    "typical_mileage": 75000,
    "typical_year": 2021,
    "remaining_life_miles": 225000,
    "down_payment_pct": 0.20,
    "loan_apr": 0.075,
    "loan_term_months": 48,
    "depreciation_years": 5,
    "salvage_value_pct": 0.10,
    "annual_maintenance_usd": 3200,
    "annual_insurance_usd": 3600,
    "maintenance_cost_per_mile": 0.08,
}

OPERATING = {
    "avg_load_unload_min_per_stop": 8,
    "avg_speed_mph_overnight": 12,
    "dispatch_overhead_min": 20,
    "return_to_depot_miles": 3,
}


def compute_bike_capacity(truck_key):
    truck = TRUCK_CANDIDATES[truck_key]
    rack = RACK_CONFIG

    usable_width = truck["cargo_width_in"] - 2 * rack["wall_clearance_in"]
    usable_length = truck["cargo_length_in"] - rack["aisle_width_in"]

    bikes_per_row = int(usable_width // rack["footprint_per_bike_width_in"])
    rows = int(usable_length // (rack["footprint_per_bike_length_in"] + rack["row_spacing_in"]))

    if truck["cargo_height_in"] >= rack["vertical_clearance_needed_in"] * 2:
        vertical_tiers = 2
    elif truck["cargo_height_in"] >= rack["vertical_clearance_needed_in"]:
        vertical_tiers = 1
    else:
        vertical_tiers = 1

    max_by_space = bikes_per_row * rows * vertical_tiers

    avg_bike_weight = (
        BIKE_SPECS["ebike"]["weight_lbs"] * BIKE_SPECS["fleet_ebike_share"]
        + BIKE_SPECS["classic"]["weight_lbs"] * BIKE_SPECS["fleet_classic_share"]
    )
    available_payload = truck["payload_lbs"] - rack["rack_weight_lbs"]
    max_by_weight = int(available_payload / avg_bike_weight)

    actual_capacity = min(max_by_space, max_by_weight)

    return {
        "truck": truck["name"],
        "cargo_volume_cuft": round(truck["cargo_length_in"] * truck["cargo_width_in"] * truck["cargo_height_in"] / 1728, 0),
        "usable_width_in": usable_width,
        "usable_length_in": usable_length,
        "bikes_per_row": bikes_per_row,
        "rows": rows,
        "vertical_tiers": vertical_tiers,
        "max_by_space": max_by_space,
        "avg_bike_weight_lbs": round(avg_bike_weight, 1),
        "available_payload_lbs": available_payload,
        "max_by_weight": max_by_weight,
        "actual_capacity": actual_capacity,
    }


def compute_fuel_cost_per_mile(truck_key):
    truck = TRUCK_CANDIDATES[truck_key]
    fuel_price = FUEL["nj_regular_per_gal_usd"] if FUEL["refuel_location"] == "NJ" else FUEL["ny_regular_per_gal_usd"]
    cost_per_mile = fuel_price / truck["mpg_city_loaded"]
    return {
        "fuel_price_per_gal": fuel_price,
        "mpg": truck["mpg_city_loaded"],
        "fuel_cost_per_mile": round(cost_per_mile, 4),
        "refuel_location": FUEL["refuel_location"],
    }


def compute_labor_cost_per_shift():
    lab = LABOR
    gross_hourly = lab["base_hourly_wage_usd"] * (1 + lab["night_shift_differential_pct"])
    loaded_hourly = gross_hourly * lab["payroll_tax_benefits_multiplier"]
    total_hours = lab["shift_duration_hours"] + lab["pre_post_shift_overhead_hours"]
    cost_per_worker = loaded_hourly * total_hours
    cost_per_truck = cost_per_worker * lab["workers_per_truck"]

    return {
        "base_hourly": lab["base_hourly_wage_usd"],
        "night_differential_pct": lab["night_shift_differential_pct"],
        "gross_hourly": round(gross_hourly, 2),
        "payroll_multiplier": lab["payroll_tax_benefits_multiplier"],
        "loaded_hourly": round(loaded_hourly, 2),
        "shift_hours": total_hours,
        "workers_per_truck": lab["workers_per_truck"],
        "cost_per_worker_per_shift": round(cost_per_worker, 2),
        "cost_per_truck_per_shift": round(cost_per_truck, 2),
    }


def compute_vehicle_cost_per_night(truck_key, method="lease"):
    truck = TRUCK_CANDIDATES[truck_key]

    if method == "lease":
        box_size = "18ft" if "18ft" in truck_key else "16ft"
        monthly = LEASE["monthly_rate_18ft_usd"] if box_size == "18ft" else LEASE["monthly_rate_16ft_usd"]
        total_monthly = monthly + LEASE["insurance_monthly_usd"]
        nightly = total_monthly / 30
        return {
            "method": "Full-Service Lease",
            "purchase_price": 0,
            "monthly_lease": monthly,
            "monthly_insurance": LEASE["insurance_monthly_usd"],
            "total_monthly": total_monthly,
            "annual_total": round(total_monthly * 12, 2),
            "nightly_vehicle_cost": round(nightly, 2),
        }
    elif method == "buy_new":
        b = BUY_NEW
        msrp = truck["msrp_usd"]
        down = msrp * b["down_payment_pct"]
        financed = msrp - down
        r = b["loan_apr"] / 12
        n = b["loan_term_months"]
        monthly_payment = financed * (r * (1 + r)**n) / ((1 + r)**n - 1)
        annual_total = (monthly_payment * 12) + b["annual_maintenance_usd"] + b["annual_insurance_usd"]
        nightly = annual_total / 365
        return {
            "method": "Buy New",
            "purchase_price": msrp,
            "down_payment": round(down, 2),
            "monthly_loan_payment": round(monthly_payment, 2),
            "loan_apr": b["loan_apr"],
            "loan_term_months": b["loan_term_months"],
            "annual_maintenance": b["annual_maintenance_usd"],
            "annual_insurance": b["annual_insurance_usd"],
            "annual_total": round(annual_total, 2),
            "nightly_vehicle_cost": round(nightly, 2),
        }
    elif method == "buy_used":
        b = BUY_USED
        price = round(truck["msrp_usd"] * b["price_pct_of_msrp"])
        down = price * b["down_payment_pct"]
        financed = price - down
        r = b["loan_apr"] / 12
        n = b["loan_term_months"]
        monthly_payment = financed * (r * (1 + r)**n) / ((1 + r)**n - 1)
        annual_total = (monthly_payment * 12) + b["annual_maintenance_usd"] + b["annual_insurance_usd"]
        nightly = annual_total / 365
        return {
            "method": f"Buy Used (~{b['typical_year']}, ~{b['typical_mileage']//1000}K mi)",
            "purchase_price": price,
            "down_payment": round(down, 2),
            "monthly_loan_payment": round(monthly_payment, 2),
            "loan_apr": b["loan_apr"],
            "loan_term_months": b["loan_term_months"],
            "annual_maintenance": b["annual_maintenance_usd"],
            "annual_insurance": b["annual_insurance_usd"],
            "annual_total": round(annual_total, 2),
            "nightly_vehicle_cost": round(nightly, 2),
            "remaining_life_miles": b["remaining_life_miles"],
            "maint_per_mile": b["maintenance_cost_per_mile"],
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_route_cost(truck_key, route_miles, n_stops, method="lease"):
    fuel = compute_fuel_cost_per_mile(truck_key)
    labor = compute_labor_cost_per_shift()
    vehicle = compute_vehicle_cost_per_night(truck_key, method)

    total_miles = route_miles + OPERATING["return_to_depot_miles"] * 2
    fuel_cost = total_miles * fuel["fuel_cost_per_mile"]

    driving_hours = total_miles / OPERATING["avg_speed_mph_overnight"]
    stop_hours = n_stops * OPERATING["avg_load_unload_min_per_stop"] / 60
    dispatch_hours = OPERATING["dispatch_overhead_min"] / 60
    total_route_hours = driving_hours + stop_hours + dispatch_hours

    return {
        "route_miles": route_miles,
        "total_miles_with_depot": round(total_miles, 1),
        "n_stops": n_stops,
        "fuel_cost": round(fuel_cost, 2),
        "labor_cost": labor["cost_per_truck_per_shift"],
        "vehicle_cost": vehicle["nightly_vehicle_cost"],
        "total_nightly_cost": round(fuel_cost + labor["cost_per_truck_per_shift"] + vehicle["nightly_vehicle_cost"], 2),
        "total_route_hours": round(total_route_hours, 2),
        "fits_in_shift": total_route_hours <= LABOR["shift_duration_hours"],
    }


def compute_fleet_cost(n_trucks, truck_key="isuzu_npr_hd_16ft", method="lease"):
    cap = compute_bike_capacity(truck_key)
    fuel = compute_fuel_cost_per_mile(truck_key)
    labor = compute_labor_cost_per_shift()
    vehicle = compute_vehicle_cost_per_night(truck_key, method)

    try:
        routes = pd.read_csv(DATA_PROCESSED / "optimal_routes.csv")
        if len(routes) >= n_trucks:
            truck_routes = routes.head(n_trucks)
            total_miles = truck_routes["miles"].sum() + n_trucks * OPERATING["return_to_depot_miles"] * 2
            total_stops = truck_routes["stops"].sum()
            total_bikes = truck_routes["bikes_moved"].sum()
        else:
            avg_miles = routes["miles"].mean()
            avg_stops = routes["stops"].mean()
            avg_bikes = routes["bikes_moved"].mean()
            total_miles = avg_miles * n_trucks + n_trucks * OPERATING["return_to_depot_miles"] * 2
            total_stops = int(avg_stops * n_trucks)
            total_bikes = int(avg_bikes * n_trucks)
    except FileNotFoundError:
        total_miles = 18 * n_trucks + n_trucks * OPERATING["return_to_depot_miles"] * 2
        total_stops = 90 * n_trucks
        total_bikes = 900 * n_trucks

    fuel_cost = total_miles * fuel["fuel_cost_per_mile"]
    labor_cost = n_trucks * labor["cost_per_truck_per_shift"]
    vehicle_cost = n_trucks * vehicle["nightly_vehicle_cost"]
    total = fuel_cost + labor_cost + vehicle_cost

    return {
        "n_trucks": n_trucks,
        "truck_model": TRUCK_CANDIDATES[truck_key]["name"],
        "bike_capacity_per_truck": cap["actual_capacity"],
        "method": method,
        "total_route_miles": round(total_miles, 1),
        "total_stops": total_stops,
        "total_bikes_moved": total_bikes,
        "fuel_cost": round(fuel_cost, 2),
        "labor_cost": round(labor_cost, 2),
        "vehicle_cost": round(vehicle_cost, 2),
        "total_nightly_cost": round(total, 2),
        "cost_per_bike_moved": round(total / max(total_bikes, 1), 2),
    }


def run_full_analysis():
    print("=" * 70)
    print("OPERATIONAL COST ANALYSIS")
    print("=" * 70)

    print("\n--- STEP 1: Bike Dimensions & Weight ---")
    avg_w = (BIKE_SPECS["ebike"]["weight_lbs"] * BIKE_SPECS["fleet_ebike_share"]
             + BIKE_SPECS["classic"]["weight_lbs"] * BIKE_SPECS["fleet_classic_share"])
    print(f"  Classic bike: {BIKE_SPECS['classic']['length_in']}\"L x {BIKE_SPECS['classic']['width_in']}\"W x {BIKE_SPECS['classic']['height_in']}\"H, {BIKE_SPECS['classic']['weight_lbs']} lbs")
    print(f"  E-bike:       {BIKE_SPECS['ebike']['length_in']}\"L x {BIKE_SPECS['ebike']['width_in']}\"W x {BIKE_SPECS['ebike']['height_in']}\"H, {BIKE_SPECS['ebike']['weight_lbs']} lbs")
    print(f"  Fleet mix:    {BIKE_SPECS['fleet_ebike_share']:.0%} e-bike, {BIKE_SPECS['fleet_classic_share']:.0%} classic")
    print(f"  Weighted avg: {avg_w:.1f} lbs/bike")

    print("\n--- STEP 2: Truck Capacity Analysis ---")
    best_truck = None
    best_cap = 0
    for key in TRUCK_CANDIDATES:
        cap = compute_bike_capacity(key)
        truck = TRUCK_CANDIDATES[key]
        print(f"\n  {cap['truck']}:")
        print(f"    Cargo: {truck['cargo_length_in']}\"L x {truck['cargo_width_in']}\"W x {truck['cargo_height_in']}\"H = {cap['cargo_volume_cuft']:.0f} cu ft")
        print(f"    Usable area (after wall clearance + aisle): {cap['usable_width_in']}\" x {cap['usable_length_in']}\"")
        print(f"    Rack layout: {cap['bikes_per_row']} bikes/row x {cap['rows']} rows x {cap['vertical_tiers']} tier(s)")
        print(f"    Max by space: {cap['max_by_space']} bikes")
        print(f"    Payload: {cap['available_payload_lbs']:,} lbs / {cap['avg_bike_weight_lbs']} lbs/bike = {cap['max_by_weight']} bikes")
        print(f"    ACTUAL CAPACITY: {cap['actual_capacity']} bikes (limited by {'weight' if cap['max_by_weight'] < cap['max_by_space'] else 'space'})")
        print(f"    Engine: {truck['engine']}")
        print(f"    CDL required: {'Yes' if truck['cdl_required'] else 'No (under 26,001 lbs GVWR)'}")
        print(f"    MSRP: ${truck['msrp_usd']:,}")
        if cap["actual_capacity"] > best_cap:
            best_cap = cap["actual_capacity"]
            best_truck = key

    print(f"\n  >> SELECTED: {TRUCK_CANDIDATES[best_truck]['name']} ({best_cap} bikes)")

    print("\n--- STEP 3: Fuel Economics ---")
    fuel = compute_fuel_cost_per_mile(best_truck)
    print(f"  Refueling in {fuel['refuel_location']}: ${fuel['fuel_price_per_gal']:.2f}/gal")
    print(f"  Truck MPG (city, loaded): {fuel['mpg']}")
    print(f"  Fuel cost per mile: ${fuel['fuel_cost_per_mile']:.4f}")
    print(f"    (= ${fuel['fuel_price_per_gal']:.2f} / {fuel['mpg']} mpg)")

    print("\n--- STEP 4: Labor Cost ---")
    labor = compute_labor_cost_per_shift()
    print(f"  Base hourly wage:        ${labor['base_hourly']:.2f}/hr")
    print(f"    Source: BLS light truck drivers NYC metro median ($25-30/hr)")
    print(f"  Night shift differential: +{labor['night_differential_pct']:.0%}")
    print(f"  Gross hourly:            ${labor['gross_hourly']:.2f}/hr")
    print(f"    (= ${labor['base_hourly']:.2f} x {1 + labor['night_differential_pct']:.2f})")
    print(f"  Payroll taxes/benefits:  x{labor['payroll_multiplier']}")
    print(f"    (FICA 7.65% + workers comp ~5% + health/benefits ~17%)")
    print(f"  Fully loaded hourly:     ${labor['loaded_hourly']:.2f}/hr")
    print(f"  Shift: {labor['shift_hours']} hrs (5hr route + 0.75hr pre/post)")
    print(f"  Workers per truck:       {labor['workers_per_truck']} (driver + loader)")
    print(f"  LABOR PER TRUCK/NIGHT:   ${labor['cost_per_truck_per_shift']:.2f}")
    print(f"    (= ${labor['loaded_hourly']:.2f}/hr x {labor['shift_hours']}hr x {labor['workers_per_truck']} workers)")

    print("\n--- STEP 5: Vehicle Cost (Lease vs. Buy New vs. Buy Used) ---")
    methods = ["lease", "buy_new", "buy_used"]
    vehicle_results = {}
    for m in methods:
        v = compute_vehicle_cost_per_night(best_truck, m)
        vehicle_results[m] = v
        print(f"\n  {v['method'].upper()}:")
        if m == "lease":
            print(f"    Monthly lease:    ${v['monthly_lease']:,}")
            print(f"    Monthly insurance:${v['monthly_insurance']}")
            print(f"    Total monthly:    ${v['total_monthly']:,}")
            print(f"    Source: Penske/Ryder commercial 36-month full-service rates")
        else:
            print(f"    Purchase price:   ${v['purchase_price']:,}")
            print(f"    Down payment:     ${v['down_payment']:,.2f} ({BUY_USED['down_payment_pct'] if 'used' in m else BUY_NEW['down_payment_pct']:.0%})")
            print(f"    Monthly payment:  ${v['monthly_loan_payment']:,.2f} ({v['loan_apr']:.1%} APR, {v['loan_term_months']}mo)")
            print(f"    Annual maint:     ${v['annual_maintenance']:,}")
            print(f"    Annual insurance: ${v['annual_insurance']:,}")
            if "remaining_life_miles" in v:
                est_years = v["remaining_life_miles"] / 6500
                print(f"    Remaining life:   ~{v['remaining_life_miles']:,} mi (~{est_years:.0f} yrs at 6,500 mi/yr)")
                print(f"    Maint per mile:   ${v['maint_per_mile']:.2f} (TMC fleet data: $0.06-$0.09)")
        print(f"    Annual total:     ${v['annual_total']:,.2f}")
        print(f"    NIGHTLY:          ${v['nightly_vehicle_cost']:.2f}")

    cheapest = min(vehicle_results.items(), key=lambda x: x[1]["nightly_vehicle_cost"])
    priciest = max(vehicle_results.items(), key=lambda x: x[1]["nightly_vehicle_cost"])
    print(f"\n  >> CHEAPEST: {cheapest[1]['method']} at ${cheapest[1]['nightly_vehicle_cost']:.2f}/night")
    print(f"     Saves ${priciest[1]['nightly_vehicle_cost'] - cheapest[1]['nightly_vehicle_cost']:.2f}/night vs {priciest[1]['method']}")

    print("\n  --- Truck Comparison (all used, buy) ---")
    for key in TRUCK_CANDIDATES:
        cap = compute_bike_capacity(key)
        v = compute_vehicle_cost_per_night(key, "buy_used")
        print(f"    {TRUCK_CANDIDATES[key]['name']:25s} | {cap['actual_capacity']} bikes | ${v['purchase_price']:,} used | ${v['nightly_vehicle_cost']:.2f}/night | {TRUCK_CANDIDATES[key]['mpg_city_loaded']} mpg")

    best_method = cheapest[0]

    print(f"\n--- STEP 6: Fleet Cost Sweep (1-8 trucks, {best_method}) ---")
    fleet_results = []
    for n in range(1, 9):
        fc = compute_fleet_cost(n, best_truck, best_method)
        fleet_results.append(fc)
        print(f"  {n} truck(s): ${fc['total_nightly_cost']:,.2f}/night | {fc['total_bikes_moved']} bikes | ${fc['cost_per_bike_moved']:.2f}/bike | {fc['total_route_miles']:.0f} mi")

    print(f"\n--- STEP 7: Cost Breakdown (4 trucks, {best_method}) ---")
    fc4 = compute_fleet_cost(4, best_truck, best_method)
    total = fc4["total_nightly_cost"]
    print(f"  Fuel:    ${fc4['fuel_cost']:>8,.2f}  ({fc4['fuel_cost']/total*100:.1f}%)")
    print(f"  Labor:   ${fc4['labor_cost']:>8,.2f}  ({fc4['labor_cost']/total*100:.1f}%)")
    print(f"  Vehicle: ${fc4['vehicle_cost']:>8,.2f}  ({fc4['vehicle_cost']/total*100:.1f}%)")
    print(f"  TOTAL:   ${total:>8,.2f}")
    print(f"  Monthly: ${total * 30:>10,.2f}")
    print(f"  Annual:  ${total * 365:>10,.2f}")

    plot_analysis(fleet_results, fc4, best_truck)

    return fleet_results, best_truck


def plot_analysis(fleet_results, fc4, truck_key):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Operational Cost Analysis — {TRUCK_CANDIDATES[truck_key]['name']}", fontsize=14, fontweight="bold")

    trucks = [r["n_trucks"] for r in fleet_results]
    costs = [r["total_nightly_cost"] for r in fleet_results]
    per_bike = [r["cost_per_bike_moved"] for r in fleet_results]

    ax1 = axes[0]
    bars = ax1.bar(trucks, costs, color="#378ADD", alpha=0.85, edgecolor="white")
    for bar, c in zip(bars, costs):
        ax1.text(bar.get_x() + bar.get_width()/2, c + 20, f"${c:,.0f}", ha="center", fontsize=8, fontweight="bold")
    ax1.set_xlabel("Number of Trucks")
    ax1.set_ylabel("Nightly Cost ($)")
    ax1.set_title("Total Nightly Cost by Fleet Size")
    ax1.set_xticks(trucks)

    total = fc4["total_nightly_cost"]
    labels = ["Fuel", "Labor", "Vehicle"]
    values = [fc4["fuel_cost"], fc4["labor_cost"], fc4["vehicle_cost"]]
    colors_pie = ["#E24B4A", "#378ADD", "#639922"]
    wedges, texts, autotexts = axes[1].pie(
        values, labels=labels, colors=colors_pie, autopct=lambda p: f"${p*total/100:,.0f}\n({p:.1f}%)",
        startangle=90, textprops={"fontsize": 9}
    )
    axes[1].set_title("Cost Breakdown (4 Trucks)")

    ax3 = axes[2]
    ax3.plot(trucks, per_bike, "o-", color="#E24B4A", linewidth=2, markersize=6)
    for t, p in zip(trucks, per_bike):
        ax3.annotate(f"${p:.2f}", (t, p), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax3.set_xlabel("Number of Trucks")
    ax3.set_ylabel("Cost per Bike Moved ($)")
    ax3.set_title("Marginal Cost Efficiency")
    ax3.set_xticks(trucks)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_operational_costs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[figure] Saved {FIGURES_DIR / '09_operational_costs.png'}")


if __name__ == "__main__":
    run_full_analysis()
