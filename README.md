# COVID-19 Big Data Analytics

A local analytics prototype that stores COVID-19 time-series data in Apache Cassandra, exposes country statistics through a Flask API, and presents charts and short-term case forecasts in a Streamlit dashboard.

## What it includes

- `data_preparation.py` selects country, date, confirmed-case, and death totals from the Our World in Data CSV and writes them to Cassandra in batches.
- `api/app.py` serves country statistics and date-filtered queries from the `covid.stats` table.
- `dashboard/dashboard.py` provides the Streamlit dashboard; `dashboard/da.py` is an alternate dashboard version.
- `ml_analysis.py` prepares date-based features, compares simple regression models, and produces case forecasts.

## Requirements

- Python 3.9 or later
- A local Cassandra node reachable at `127.0.0.1`
- The Our World in Data file named `owid-covid-data.csv` in the project root (not included in the public repository)

Install the Python packages:

```bash
python -m pip install -r requirements.txt
```

## Create the Cassandra table

Run this in `cqlsh`:

```sql
CREATE KEYSPACE IF NOT EXISTS covid
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE covid;

CREATE TABLE IF NOT EXISTS stats (
    country text,
    date date,
    total_cases int,
    total_deaths int,
    PRIMARY KEY ((country), date)
);
```

## Run locally

From the project root, load the CSV into Cassandra:

```bash
python data_preparation.py
```

Start the API in one terminal and the dashboard in another:

```bash
python api/app.py
```

```bash
streamlit run dashboard/dashboard.py
```

The API listens on `http://127.0.0.1:5000`, which is the URL configured by the dashboard.

## Data and artifacts

Large datasets, local database contents, and generated outputs are kept outside version control. Obtain the CSV separately and place it at the path shown above.
