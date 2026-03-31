import requests
import zipfile
import io
import sys
from pathlib import Path
from config import TRIP_BASE_URL, TRIP_FILE_TEMPLATE, TRIP_MONTHS, DATA_RAW


def download_month(month: str) -> None:
    filename = TRIP_FILE_TEMPLATE.format(month=month)
    url = f"{TRIP_BASE_URL}/{filename}"
    dest = DATA_RAW / filename

    if dest.exists():
        print(f"[skip] {filename} already exists")
        return

    print(f"[download] {url}")
    print(f"  This file may be 200-700 MB. Please be patient...")

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1024 / 1024
                print(f"\r  {mb:.0f} MB ({pct:.1f}%)", end="", flush=True)
    print()
    print(f"[saved] {dest}")


def extract_csvs(month: str) -> None:
    filename = TRIP_FILE_TEMPLATE.format(month=month)
    zippath = DATA_RAW / filename

    if not zippath.exists():
        print(f"[error] {zippath} not found")
        return

    print(f"[extract] {zippath.name}")
    with zipfile.ZipFile(zippath, "r") as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        for csv_name in csv_files:
            dest = DATA_RAW / Path(csv_name).name
            if dest.exists():
                print(f"  [skip] {dest.name} already exists")
                continue
            print(f"  [extracting] {csv_name}")
            zf.extract(csv_name, DATA_RAW)
            extracted = DATA_RAW / csv_name
            if extracted != dest:
                extracted.rename(dest)
    print(f"[done] {len(csv_files)} CSV(s) from {month}")


def main():
    months = sys.argv[1:] if len(sys.argv) > 1 else TRIP_MONTHS

    for month in months:
        download_month(month)
        extract_csvs(month)

    csv_count = len(list(DATA_RAW.glob("*.csv")))
    print(f"\nTotal CSVs in {DATA_RAW}: {csv_count}")


if __name__ == "__main__":
    main()
