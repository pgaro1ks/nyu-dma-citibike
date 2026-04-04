import sys
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PROCESSED, FIGURES_DIR, MORNING_PEAK

sns.set_theme(style="whitegrid", palette="muted")


PRICING = {
    "annual_membership_usd": 239.00,
    "single_ride_unlock_usd": 4.99,
    "ebike_per_min_member_usd": 0.27,
    "ebike_per_min_casual_usd": 0.41,
    "classic_overage_per_min_usd": 0.27,
}

SYSTEM_SPLITS = {
    "member_share": 0.824,
    "casual_share": 0.176,
    "ebike_share": 0.706,
    "classic_share": 0.294,
}

MEMBER_ECONOMICS = {
    "annual_fee_usd": 239.00,
    "avg_trips_per_member_per_year": 200,
    "ebike_avg_ride_min": 14,
    "classic_avg_ride_min": 18,
    "free_classic_min": 45,
    "free_ebike_min_per_year": 60,
}

VOT = {
    "nyc_median_household_income_usd": 75000,
    "work_hours_per_year": 2080,
    "usdot_local_surface_pct": 0.50,
    "avg_delay_minutes_failed_trip": 12,
}

CHURN = {
    "annual_churn_rate_baseline": 0.15,
    "failed_trips_to_double_churn": 20,
    "member_lifetime_years": 3.5,
}


def compute_blended_revenue_per_trip():
    me = MEMBER_ECONOMICS
    base_fee_per_trip = me["annual_fee_usd"] / me["avg_trips_per_member_per_year"]

    ebike_share = SYSTEM_SPLITS["ebike_share"]
    classic_share = SYSTEM_SPLITS["classic_share"]

    ebike_surcharge = max(0, me["ebike_avg_ride_min"] - (me["free_ebike_min_per_year"] / me["avg_trips_per_member_per_year"])) * PRICING["ebike_per_min_member_usd"]
    classic_surcharge = max(0, me["classic_avg_ride_min"] - me["free_classic_min"]) * PRICING["classic_overage_per_min_usd"]

    member_rev = base_fee_per_trip + ebike_share * ebike_surcharge + classic_share * classic_surcharge

    casual_base = PRICING["single_ride_unlock_usd"]
    casual_ebike_surcharge = me["ebike_avg_ride_min"] * PRICING["ebike_per_min_casual_usd"]
    casual_classic_surcharge = 0
    casual_rev = casual_base + ebike_share * casual_ebike_surcharge + classic_share * casual_classic_surcharge

    blended = SYSTEM_SPLITS["member_share"] * member_rev + SYSTEM_SPLITS["casual_share"] * casual_rev

    return {
        "member_rev_per_trip": round(member_rev, 4),
        "casual_rev_per_trip": round(casual_rev, 4),
        "blended_rev_per_trip": round(blended, 4),
        "member_base_fee_per_trip": round(base_fee_per_trip, 4),
        "member_ebike_surcharge": round(ebike_surcharge, 4),
        "casual_base": round(casual_base, 4),
        "casual_ebike_surcharge": round(casual_ebike_surcharge, 4),
    }


def compute_value_of_time_cost():
    hourly_income = VOT["nyc_median_household_income_usd"] / VOT["work_hours_per_year"]
    vtts_per_hour = hourly_income * VOT["usdot_local_surface_pct"]
    vtts_per_minute = vtts_per_hour / 60
    delay_cost = vtts_per_minute * VOT["avg_delay_minutes_failed_trip"]

    return {
        "hourly_income": round(hourly_income, 2),
        "vtts_per_hour": round(vtts_per_hour, 2),
        "vtts_per_minute": round(vtts_per_minute, 4),
        "avg_delay_min": VOT["avg_delay_minutes_failed_trip"],
        "delay_cost_per_failed_trip": round(delay_cost, 2),
    }


def compute_churn_cost():
    ch = CHURN
    me = MEMBER_ECONOMICS
    ltv = me["annual_fee_usd"] * ch["member_lifetime_years"]
    incremental_churn_per_failure = ch["annual_churn_rate_baseline"] / ch["failed_trips_to_double_churn"]
    churn_cost_per_failure = incremental_churn_per_failure * ltv

    return {
        "member_ltv": round(ltv, 2),
        "baseline_annual_churn": ch["annual_churn_rate_baseline"],
        "incremental_churn_per_failure": round(incremental_churn_per_failure, 4),
        "churn_cost_per_failed_trip": round(churn_cost_per_failure, 2),
    }


def compute_unmet_trips():
    target = pd.read_parquet(DATA_PROCESSED / "rebalancing_targets.parquet")
    stations = pd.read_parquet(DATA_PROCESSED / "station_demand_stats.parquet")

    target = target.merge(
        stations[["station_id", "mean_departures", "mean_arrivals"]],
        on="station_id",
        how="left",
        suffixes=("", "_stats"),
    )

    if "mean_departures_stats" in target.columns:
        target["mean_departures"] = target["mean_departures"].fillna(target["mean_departures_stats"])
        target["mean_arrivals"] = target["mean_arrivals"].fillna(target["mean_arrivals_stats"])

    unmet_dep = np.maximum(
        target["mean_departures"] - target["estimated_midnight_fill"], 0
    ).sum()
    unmet_arr = np.maximum(
        target["mean_arrivals"] - (target["capacity"] - target["estimated_midnight_fill"]), 0
    ).sum()

    return {
        "unmet_departures": round(unmet_dep, 1),
        "unmet_arrivals": round(unmet_arr, 1),
        "total_unmet_trips": round(unmet_dep + unmet_arr, 1),
    }


def compute_total_cost(rev, vot, churn, unmet):
    direct_revenue_loss = unmet["total_unmet_trips"] * rev["blended_rev_per_trip"]
    time_cost = unmet["total_unmet_trips"] * vot["delay_cost_per_failed_trip"]
    churn_risk = unmet["total_unmet_trips"] * SYSTEM_SPLITS["member_share"] * churn["churn_cost_per_failed_trip"]
    total_daily = direct_revenue_loss + time_cost + churn_risk

    return {
        "direct_revenue_loss_daily": round(direct_revenue_loss, 2),
        "time_cost_daily": round(time_cost, 2),
        "churn_risk_daily": round(churn_risk, 2),
        "total_daily": round(total_daily, 2),
        "total_monthly": round(total_daily * 30, 2),
        "total_annual": round(total_daily * 365, 2),
        "per_trip_breakdown": {
            "revenue": round(rev["blended_rev_per_trip"], 2),
            "time_cost": round(vot["delay_cost_per_failed_trip"], 2),
            "churn_risk": round(SYSTEM_SPLITS["member_share"] * churn["churn_cost_per_failed_trip"], 2),
            "total_per_trip": round(
                rev["blended_rev_per_trip"]
                + vot["delay_cost_per_failed_trip"]
                + SYSTEM_SPLITS["member_share"] * churn["churn_cost_per_failed_trip"],
                2,
            ),
        },
    }


def sensitivity_analysis(rev, vot, churn, unmet):
    scenarios = []
    base_per_trip = (
        rev["blended_rev_per_trip"]
        + vot["delay_cost_per_failed_trip"]
        + SYSTEM_SPLITS["member_share"] * churn["churn_cost_per_failed_trip"]
    )

    params = {
        "Avg delay (min)": {
            "base": VOT["avg_delay_minutes_failed_trip"],
            "range": [5, 8, 12, 15, 20],
            "calc": lambda v: unmet["total_unmet_trips"] * (
                rev["blended_rev_per_trip"]
                + (vot["vtts_per_hour"] / 60) * v
                + SYSTEM_SPLITS["member_share"] * churn["churn_cost_per_failed_trip"]
            ),
        },
        "Trips/member/year": {
            "base": MEMBER_ECONOMICS["avg_trips_per_member_per_year"],
            "range": [100, 150, 200, 300, 400],
            "calc": lambda v: unmet["total_unmet_trips"] * (
                SYSTEM_SPLITS["member_share"] * (MEMBER_ECONOMICS["annual_fee_usd"] / v + SYSTEM_SPLITS["ebike_share"] * rev["member_ebike_surcharge"])
                + SYSTEM_SPLITS["casual_share"] * rev["casual_rev_per_trip"]
                + vot["delay_cost_per_failed_trip"]
                + SYSTEM_SPLITS["member_share"] * (CHURN["annual_churn_rate_baseline"] / CHURN["failed_trips_to_double_churn"]) * (MEMBER_ECONOMICS["annual_fee_usd"] * CHURN["member_lifetime_years"])
            ),
        },
        "Member lifetime (yrs)": {
            "base": CHURN["member_lifetime_years"],
            "range": [2.0, 2.5, 3.5, 5.0, 7.0],
            "calc": lambda v: unmet["total_unmet_trips"] * (
                rev["blended_rev_per_trip"]
                + vot["delay_cost_per_failed_trip"]
                + SYSTEM_SPLITS["member_share"] * (CHURN["annual_churn_rate_baseline"] / CHURN["failed_trips_to_double_churn"]) * (MEMBER_ECONOMICS["annual_fee_usd"] * v)
            ),
        },
        "Churn doubling threshold": {
            "base": CHURN["failed_trips_to_double_churn"],
            "range": [5, 10, 20, 40, 60],
            "calc": lambda v: unmet["total_unmet_trips"] * (
                rev["blended_rev_per_trip"]
                + vot["delay_cost_per_failed_trip"]
                + SYSTEM_SPLITS["member_share"] * (CHURN["annual_churn_rate_baseline"] / v) * (MEMBER_ECONOMICS["annual_fee_usd"] * CHURN["member_lifetime_years"])
            ),
        },
    }

    for name, p in params.items():
        for val in p["range"]:
            daily = p["calc"](val)
            scenarios.append({
                "parameter": name,
                "value": val,
                "is_base": abs(val - p["base"]) < 0.01,
                "daily_cost": round(daily, 2),
                "monthly_cost": round(daily * 30, 2),
            })

    return pd.DataFrame(scenarios)


def plot_cost_breakdown(total, unmet, sensitivity_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cost of Inaction — Methodology & Sensitivity", fontsize=14, fontweight="bold")

    labels = ["Direct Revenue\nLoss", "Rider Time\nCost (USDOT VOT)", "Subscriber\nChurn Risk"]
    values = [total["direct_revenue_loss_daily"], total["time_cost_daily"], total["churn_risk_daily"]]
    colors = ["#E24B4A", "#378ADD", "#639922"]
    bars = axes[0].bar(labels, values, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 30, f"${v:,.0f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Daily cost ($)")
    axes[0].set_title(f"Daily Cost Breakdown (Total: ${total['total_daily']:,.0f})")

    pt = total["per_trip_breakdown"]
    components = [pt["revenue"], pt["time_cost"], pt["churn_risk"]]
    comp_labels = ["Revenue", "Time (VOT)", "Churn"]
    bottom = 0
    for val, label, color in zip(components, comp_labels, colors):
        axes[1].bar("Per Failed Trip", val, bottom=bottom, color=color, alpha=0.85, label=f"{label}: ${val:.2f}", edgecolor="white")
        if val > 0.3:
            axes[1].text(0, bottom + val / 2, f"${val:.2f}", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        bottom += val
    axes[1].set_ylabel("Cost per failed trip ($)")
    axes[1].set_title(f"Per-Trip Composition (${pt['total_per_trip']:.2f}/trip)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylim(0, bottom * 1.15)

    for param in sensitivity_df["parameter"].unique():
        subset = sensitivity_df[sensitivity_df["parameter"] == param]
        axes[2].plot(subset["value"], subset["daily_cost"], "o-", label=param, markersize=5)
        base_row = subset[subset["is_base"]]
        if not base_row.empty:
            axes[2].plot(base_row["value"].values[0], base_row["daily_cost"].values[0], "D", color="red", markersize=8, zorder=5)
    axes[2].set_ylabel("Daily cost ($)")
    axes[2].set_xlabel("Parameter value")
    axes[2].set_title("Sensitivity to Key Assumptions")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].axhline(total["total_daily"], color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_cost_of_inaction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved {FIGURES_DIR / '08_cost_of_inaction.png'}")


def main():
    print("=" * 70)
    print("COST OF INACTION ANALYSIS")
    print("=" * 70)

    print("\n--- STEP 1: Blended Revenue Per Trip ---")
    rev = compute_blended_revenue_per_trip()
    print(f"  Member base fee/trip:    ${rev['member_base_fee_per_trip']:.4f}")
    print(f"    (= $239 / 200 trips/yr)")
    print(f"  Member e-bike surcharge: ${rev['member_ebike_surcharge']:.4f}")
    print(f"    (= max(0, 14min - 60min/200trips) * $0.27/min)")
    print(f"  Member total/trip:       ${rev['member_rev_per_trip']:.2f}")
    print(f"  Casual base unlock:      ${rev['casual_base']:.2f}")
    print(f"  Casual e-bike surcharge: ${rev['casual_ebike_surcharge']:.2f}")
    print(f"    (= 14min * $0.41/min)")
    print(f"  Casual total/trip:       ${rev['casual_rev_per_trip']:.2f}")
    print(f"  Blended (82.4%/17.6%):   ${rev['blended_rev_per_trip']:.2f}")

    print("\n--- STEP 2: USDOT Value of Travel Time ---")
    vot = compute_value_of_time_cost()
    print(f"  NYC median HH income:    ${VOT['nyc_median_household_income_usd']:,}")
    print(f"  Hourly income:           ${vot['hourly_income']:.2f}")
    print(f"  VTTS (50% of hourly):    ${vot['vtts_per_hour']:.2f}/hr")
    print(f"  VTTS per minute:         ${vot['vtts_per_minute']:.4f}")
    print(f"  Avg delay per failure:   {vot['avg_delay_min']} min")
    print(f"  Delay cost/failed trip:  ${vot['delay_cost_per_failed_trip']:.2f}")
    print(f"    (= ${vot['vtts_per_minute']:.4f}/min * {vot['avg_delay_min']}min)")

    print("\n--- STEP 3: Subscriber Churn Risk ---")
    churn = compute_churn_cost()
    print(f"  Member LTV:              ${churn['member_ltv']:.2f}")
    print(f"    (= $239/yr * {CHURN['member_lifetime_years']} yrs)")
    print(f"  Baseline annual churn:   {churn['baseline_annual_churn']:.0%}")
    print(f"  Failures to double churn:{CHURN['failed_trips_to_double_churn']}")
    print(f"  Incremental churn/fail:  {churn['incremental_churn_per_failure']:.4f}")
    print(f"    (= {churn['baseline_annual_churn']:.0%} / {CHURN['failed_trips_to_double_churn']} failures)")
    print(f"  Churn cost/failed trip:  ${churn['churn_cost_per_failed_trip']:.2f}")
    print(f"    (= {churn['incremental_churn_per_failure']:.4f} * ${churn['member_ltv']:.2f})")

    print("\n--- STEP 4: Unmet Trips (from imbalance model) ---")
    unmet = compute_unmet_trips()
    print(f"  Unmet departures/morning: {unmet['unmet_departures']}")
    print(f"  Unmet arrivals/morning:   {unmet['unmet_arrivals']}")
    print(f"  Total unmet trips:        {unmet['total_unmet_trips']}")

    print("\n--- STEP 5: Total Cost of Inaction ---")
    total = compute_total_cost(rev, vot, churn, unmet)
    pt = total["per_trip_breakdown"]
    print(f"\n  Per failed trip:")
    print(f"    Direct revenue:        ${pt['revenue']:.2f}")
    print(f"    Time cost (VOT):       ${pt['time_cost']:.2f}")
    print(f"    Churn risk:            ${pt['churn_risk']:.2f}")
    print(f"    TOTAL per trip:        ${pt['total_per_trip']:.2f}")
    print(f"\n  Daily (x {unmet['total_unmet_trips']:.0f} unmet trips):")
    print(f"    Direct revenue loss:   ${total['direct_revenue_loss_daily']:,.2f}")
    print(f"    Rider time cost:       ${total['time_cost_daily']:,.2f}")
    print(f"    Churn risk cost:       ${total['churn_risk_daily']:,.2f}")
    print(f"    TOTAL daily:           ${total['total_daily']:,.2f}")
    print(f"    Monthly:               ${total['total_monthly']:,.2f}")
    print(f"    Annual:                ${total['total_annual']:,.2f}")

    print("\n--- STEP 6: Sensitivity Analysis ---")
    sens = sensitivity_analysis(rev, vot, churn, unmet)
    print(sens.to_string(index=False))

    plot_cost_breakdown(total, unmet, sens)

    results = {
        "revenue": rev,
        "vot": vot,
        "churn": churn,
        "unmet": unmet,
        "total": total,
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Blended revenue/trip:    ${rev['blended_rev_per_trip']:.2f} (from actual 82/18 member/casual split)")
    print(f"  USDOT time cost/trip:    ${vot['delay_cost_per_failed_trip']:.2f} (12-min delay at $18.03/hr)")
    print(f"  Churn risk/trip:         ${pt['churn_risk']:.2f} (member-weighted)")
    print(f"  Total cost/failed trip:  ${pt['total_per_trip']:.2f}")
    print(f"  x {unmet['total_unmet_trips']:.0f} unmet trips/morning")
    print(f"  = ${total['total_daily']:,.2f}/day | ${total['total_monthly']:,.2f}/month | ${total['total_annual']:,.2f}/year")

    return results


if __name__ == "__main__":
    main()
