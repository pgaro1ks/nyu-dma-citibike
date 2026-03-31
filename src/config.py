from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_LIVE = PROJECT_ROOT / "data" / "live"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXCEL_DIR = PROJECT_ROOT / "excel"

for d in [DATA_RAW, DATA_PROCESSED, DATA_LIVE, FIGURES_DIR, EXCEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRIP_MONTHS = ["202409", "202410", "202411"]

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
    "truck_capacity_bikes": 20,
    "fixed_cost_per_truck_per_night_usd": 450.0,
    "variable_cost_per_mile_usd": 3.50,
    "driver_hourly_wage_usd": 35.0,
    "overtime_multiplier": 1.5,
    "overtime_threshold_hours": 8.0,
    "fuel_cost_per_gallon_usd": 3.80,
    "truck_mpg": 8.0,
    "avg_load_unload_minutes_per_stop": 12.0,
    "avg_speed_mph_overnight": 15.0,
}

REVENUE_MODEL = {
    "revenue_per_trip_usd": 4.50,
    "annual_membership_daily_equiv_usd": 0.49,
    "single_ride_price_usd": 4.49,
    "day_pass_price_usd": 19.00,
    "ebike_unlock_fee_usd": 1.00,
}

OPPORTUNITY_COST = {
    "lost_trip_revenue_usd": 4.50,
    "customer_dissatisfaction_penalty_usd": 2.00,
    "brand_damage_multiplier": 1.25,
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
