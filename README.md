# Citi Bike Overnight Rebalancing Optimization

**NYU Stern — B60.2350 Decision Models and Analytics (Prof. Juran)**
**Term Project — Spring 2026**

## Problem

Citi Bike stations become imbalanced over the course of each day: commuter-heavy stations in Midtown fill up while residential stations in neighborhoods empty out. Overnight, Lyft deploys trucks to physically redistribute bikes so that morning rush-hour demand can be met. The question is: **how many trucks, visiting which stations, moving how many bikes, at what cost?**

## Approach

| Phase | Description | Course Techniques |
|-------|-------------|-------------------|
| 1 | Data acquisition & demand profiling | Data analysis, distribution fitting |
| 2 | Imbalance analysis & target inventory | Network flow concepts |
| 3 | Truck routing optimization | Solver (LP/IP), network models |
| 4 | Monte Carlo simulation | Crystal Ball, stochastic demand |
| 5 | Sensitivity & efficient frontier | SolverTable, multi-objective optimization |
| 6 | Deliverables | Excel workbook, presentation, writeup |

## Data Sources

- **Historical trip data**: [Citi Bike S3 bucket](https://s3.amazonaws.com/tripdata/index.html) (Sep–Nov 2024)
- **Live station feed**: [GBFS station_status](https://gbfs.citibikenyc.com/gbfs/en/station_status.json)
- **Station metadata**: [GBFS station_information](https://gbfs.citibikenyc.com/gbfs/en/station_information.json)

## Cost Model

The optimization incorporates real-world cost parameters:

- **Fixed cost per truck per night**: driver wages + vehicle lease amortization
- **Variable cost per mile**: fuel, wear, tolls
- **Opportunity cost of unmet demand**: lost revenue per empty/full station event
- **Labor overtime**: premium rates for shifts exceeding 8 hours

## Repo Structure

```
src/                    Python pipeline scripts
  config.py             Cost parameters and study area definition
  01_download_trips.py  Fetch trip CSVs from S3
  02_station_meta.py    Fetch and cache station metadata from GBFS
  03_demand_profile.py  Compute per-station hourly demand distributions
  04_imbalance.py       Midnight inventory vs. morning target analysis
  05_optimize.py        Truck routing optimization model
  06_simulate.py        Monte Carlo simulation wrapper
  07_sensitivity.py     SolverTable-style sweep
  08_figures.py         Generate all matplotlib figures
data/
  raw/                  Downloaded CSVs (gitignored)
  processed/            Cleaned parquet files (gitignored)
  live/                 GBFS snapshots (gitignored)
figures/                Generated charts and maps
excel/                  Final Excel workbook
docs/                   LaTeX writeup and presentation
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn scipy requests openpyxl pyarrow
python src/01_download_trips.py
python src/02_station_meta.py
python src/03_demand_profile.py
```
