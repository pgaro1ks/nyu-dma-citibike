from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_LIVE = PROJECT_ROOT / "data" / "live"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXCEL_DIR = PROJECT_ROOT / "excel"

for d in [DATA_RAW, DATA_PROCESSED, DATA_LIVE, FIGURES_DIR, EXCEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRIP_MONTHS = [
    "202504", "202505", "202506", "202507", "202508", "202509",
    "202510", "202511", "202512", "202601", "202602", "202503",
]

TRIP_BASE_URL = "https://s3.amazonaws.com/tripdata"
TRIP_FILE_TEMPLATE = "{month}-citibike-tripdata.zip"

GBFS_STATION_STATUS = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
GBFS_STATION_INFO = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"

STUDY_AREA = {
    "name": "Manhattan below 59th St",
    "lat_min": 40.700,
    "lat_max": 40.770,
    "lon_min": -74.020,
    "lon_max": -73.970,
}

REBALANCING_WINDOW = {
    "start_hour": 0,
    "end_hour": 5,
    "duration_hours": 5,
}

MORNING_PEAK = {
    "start_hour": 7,
    "end_hour": 9,
}

COST_MODEL = {
    "truck_capacity_bikes": 48,                        # Isuzu NPR-HD 18' vertical wheel-hang rack
    "fixed_cost_per_truck_per_night_usd": 44.16,       # Buy used ~2021 Isuzu NPR-HD 18' ($40K, 7.5% APR 48mo)
    "variable_cost_per_mile_usd": 0.0,                 # Fuel handled separately below
    "driver_hourly_wage_usd": 81.54,                   # 2 workers × $40.77/hr loaded ($28 base + 12% night + 30% burden)
    "overtime_multiplier": 1.5,
    "overtime_threshold_hours": 8.0,
    "fuel_cost_per_gallon_usd": 3.30,                  # NJ regular (AAA current)
    "truck_mpg": 8.5,                                  # Isuzu NPR-HD 18' city loaded
    "avg_load_unload_minutes_per_stop": 8.0,           # 2-worker crew
    "avg_speed_mph_overnight": 12.0,                   # NYC overnight with loading stops
}

REVENUE_MODEL = {
    "revenue_per_trip_usd": 4.78,                      # Blended from 82.4% member / 17.6% casual actual split
    "annual_membership_usd": 239.00,
    "annual_membership_daily_equiv_usd": 0.65,          # $239 / 365
    "single_ride_price_usd": 4.99,                      # Casual unlock fee
    "day_pass_price_usd": 19.00,
    "ebike_unlock_fee_usd": 1.00,
    "ebike_per_min_member_usd": 0.27,
    "ebike_per_min_casual_usd": 0.41,
    "member_share": 0.824,
    "casual_share": 0.176,
}

OPPORTUNITY_COST = {
    "lost_trip_revenue_usd": 4.78,                     # Blended revenue per trip (from cost_of_inaction.py)
    "customer_dissatisfaction_penalty_usd": 8.78,      # USDOT VOT $3.61 + churn risk $5.17 (from cost_of_inaction.py)
    "brand_damage_multiplier": 1.0,                    # Soft costs already included in dissatisfaction penalty
}

TARGET_FILL_RATIO = {
    "morning_min": 0.30,
    "morning_max": 0.70,
    "ideal": 0.50,
}

SIMULATION = {
    "n_trials": 5000,
    "random_seed": 42,
}

MAX_TRUCKS = 10
MAX_STATIONS = 40
