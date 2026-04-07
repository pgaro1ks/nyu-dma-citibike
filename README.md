# Citi Bike Overnight Rebalancing Optimization

**NYU Stern — B60.2350 Decision Models and Analytics (Prof. Juran)**
**Term Project — Spring 2026**

## Problem

Citi Bike stations become imbalanced over the course of each day: commuter-heavy stations in Midtown fill up while residential stations empty out. Overnight, trucks physically redistribute bikes so that morning rush-hour demand can be met. The question is: **how many trucks, visiting which stations, moving how many bikes, at what cost?**

## Approach

| Phase | Script | Description | Techniques |
|-------|--------|-------------|------------|
| 0 | `01_download_trips.py` | Fetch 12 months of trip CSVs from S3 | Data acquisition |
| 0 | `02_station_meta.py` | Station metadata from GBFS (capacity, lat/lon) | API integration |
| 1 | `03_demand_profile.py` | Hourly demand distributions per station | Poisson/Normal fitting |
| 2 | `04_imbalance.py` | Midnight inventory vs. morning target analysis | Network flow concepts |
| 3 | `05_optimize.py` | Nearest-neighbor TSP + 2-opt truck routing | Combinatorial optimization |
| 4 | `06_simulate.py` | Monte Carlo simulation (5,000 trials) | Stochastic modeling |
| 5 | `07_sensitivity.py` | Efficient frontier + tornado diagram | Multi-objective optimization |
| 6 | `08_build_workbook.py` | Excel deliverable with 3,100+ formulas | Workbook automation |
| — | `cost_of_inaction.py` | USDOT VOT + churn risk cost model | Economic analysis |
| — | `operational_costs.py` | Truck specs, labor, fuel, fleet economics | Operations research |

## Data

- **Historical trips**: 44.6 million trips, April 2025 – March 2026 (12 months), from the [Citi Bike S3 bucket](https://s3.amazonaws.com/tripdata/index.html)
- **Live station feed**: [GBFS station_status](https://gbfs.citibikenyc.com/gbfs/en/station_status.json) and [station_information](https://gbfs.citibikenyc.com/gbfs/en/station_information.json)
- **Study area**: Manhattan below 59th St (386 stations)
- **Morning peak**: 7:00–9:00 AM

## Cost Models

### Cost of Inaction (~$13.50 per failed trip)

Three mathematically derived components:

- **Direct revenue loss** ($4.73): Blended from actual 82.4% member / 17.6% casual split, including e-bike surcharges
- **USDOT Value of Travel Time** ($3.61): NYC median household income → $18.03/hr VTTS, 12-minute average delay
- **Subscriber churn risk** ($5.17): $836.50 LTV × 0.75% incremental churn per failure, member-weighted

At ~922 unmet trips per morning without rebalancing: **$12,457/day | $374K/month | $4.5M/year**

### Operational Costs (~$2,088/night for 4 trucks)

- **Truck**: Isuzu NPR-HD 18' box truck, 48 bikes (vertical wheel-hang rack), no CDL required
- **Acquisition**: Buy used (~2021, ~75K miles, ~$40K) — cheapest at $44.16/night vs $81.67 lease
- **Fuel**: NJ refueling at $3.30/gal, 8.5 MPG loaded → $0.39/mile
- **Labor**: 2-worker crew at $40.77/hr loaded ($28 base + 12% night + 30% burden), $469/truck/night
- **Cost structure**: Labor 89%, vehicle 5%, fuel 6%

## Key Results

- **Net benefit**: ~$10,000/day (cost of inaction minus operational cost)
- **Service level**: Improves from ~70% (no rebalancing) to ~95%+ (4 trucks)
- **Monte Carlo**: 5,000 trials confirm positive net benefit in the vast majority of scenarios
- **Most sensitive parameter**: Crew wage rate

## Repo Structure

```
src/
  config.py               Central parameters (costs, study area, simulation settings)
  01_download_trips.py     Fetch trip CSVs from S3
  02_station_meta.py       Station metadata from GBFS
  03_demand_profile.py     Hourly demand distributions
  04_imbalance.py          Midnight inventory vs. morning target
  05_optimize.py           TSP truck routing (NN + 2-opt)
  06_simulate.py           Monte Carlo simulation
  07_sensitivity.py        Efficient frontier + tornado
  08_build_workbook.py     Excel workbook generator
  cost_of_inaction.py      Revenue + VOT + churn cost model
  operational_costs.py     Truck specs, labor, fuel, fleet analysis
data/
  raw/                     Downloaded trip CSVs (gitignored)
  processed/               Cleaned parquet files (gitignored)
  live/                    GBFS snapshots (gitignored)
figures/                   Generated matplotlib charts (01–09)
excel/                     Final Excel workbook
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn scipy requests openpyxl pyarrow

# Run the full pipeline in order:
python src/01_download_trips.py
python src/02_station_meta.py
python src/03_demand_profile.py
python src/04_imbalance.py
python src/05_optimize.py
python src/06_simulate.py
python src/07_sensitivity.py
python src/08_build_workbook.py

# Standalone analysis scripts (run after 04_imbalance.py):
python src/cost_of_inaction.py
python src/operational_costs.py
```

## Limitations & Future Work

- Citi Bike already operates rebalancing trucks — this model provides a framework to evaluate optimal fleet sizing and routing, not to claim the concept is novel
- Depot location optimization (assumed generic; real depot placement would reduce deadhead miles)
- Evening rebalancing (second pass before evening commute)
- Dynamic real-time rebalancing using live GBFS feed
- E-bike battery/charging logistics
- Seasonal fleet size differentiation
