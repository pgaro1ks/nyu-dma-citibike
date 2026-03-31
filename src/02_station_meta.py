import json
import requests
import pandas as pd
from config import (
    GBFS_STATION_INFO,
    GBFS_STATION_STATUS,
    DATA_LIVE,
    DATA_PROCESSED,
    STUDY_AREA,
)


def fetch_station_info() -> pd.DataFrame:
    print("[fetch] Station information from GBFS")
    resp = requests.get(GBFS_STATION_INFO)
    resp.raise_for_status()
    data = resp.json()

    with open(DATA_LIVE / "station_information.json", "w") as f:
        json.dump(data, f, indent=2)

    stations = data["data"]["stations"]
    df = pd.DataFrame(stations)

    df = df[["station_id", "name", "short_name", "lat", "lon", "capacity", "region_id"]]
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["capacity"] = df["capacity"].astype(int)

    print(f"  Total stations in system: {len(df)}")
    return df


def fetch_station_status() -> pd.DataFrame:
    print("[fetch] Station status (live snapshot) from GBFS")
    resp = requests.get(GBFS_STATION_STATUS)
    resp.raise_for_status()
    data = resp.json()

    with open(DATA_LIVE / "station_status.json", "w") as f:
        json.dump(data, f, indent=2)

    stations = data["data"]["stations"]
    df = pd.DataFrame(stations)

    keep_cols = [
        "station_id",
        "num_bikes_available",
        "num_ebikes_available",
        "num_bikes_disabled",
        "num_docks_available",
        "num_docks_disabled",
        "is_installed",
        "is_renting",
        "is_returning",
        "last_reported",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available]

    print(f"  Stations with status: {len(df)}")
    return df


def filter_study_area(df: pd.DataFrame) -> pd.DataFrame:
    sa = STUDY_AREA
    mask = (
        (df["lat"] >= sa["lat_min"])
        & (df["lat"] <= sa["lat_max"])
        & (df["lon"] >= sa["lon_min"])
        & (df["lon"] <= sa["lon_max"])
        & (df["capacity"] > 0)
    )
    filtered = df[mask].copy()
    print(f"  Stations in study area ({sa['name']}): {len(filtered)}")
    return filtered


def main():
    info_df = fetch_station_info()

    study_df = filter_study_area(info_df)

    status_df = fetch_station_status()

    merged = study_df.merge(status_df, on="station_id", how="left")

    merged.to_parquet(DATA_PROCESSED / "stations_study_area.parquet", index=False)
    merged.to_csv(DATA_PROCESSED / "stations_study_area.csv", index=False)

    print(f"\n[saved] {len(merged)} study-area stations")
    print(f"  File: {DATA_PROCESSED / 'stations_study_area.parquet'}")
    print(f"\nSample:")
    print(merged[["name", "lat", "lon", "capacity", "num_bikes_available"]].head(10).to_string())

    print(f"\nCapacity stats:")
    print(f"  Total docks: {merged['capacity'].sum()}")
    print(f"  Mean capacity: {merged['capacity'].mean():.1f}")
    print(f"  Min: {merged['capacity'].min()}, Max: {merged['capacity'].max()}")


if __name__ == "__main__":
    main()
