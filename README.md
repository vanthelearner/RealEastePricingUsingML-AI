# Dubai Real Estate Transaction Value Prediction (ML) + Mispricing Explorer



---

## What’s in this project

### Main objectives
- **Predict** transaction value from property/transaction features (size, bedrooms, area, off-plan status, nearest amenities, etc.)
- **Compare** several model families and report generalization performance
- **Explore** potential mispricing by comparing predicted vs. observed values in a hold-out set

### Models compared (4)
1. **Linear Regression** (simple features baseline)
2. **Linear Regression (extended)** with one-hot encoded `AREA_EN`
3. **Random Forest Regressor**
4. **Neural Network (MLP)**

### Evaluation
The notebook reports:
- R² (train/test) + **R² gap** (overfitting signal)
- RMSE (train/test)
- MAE (train/test)
- Basic fit/predict timing (useful for “accuracy vs cost” discussion)

---

## Repository structure

```text
.
├── notebooks/
│   └── Real_Estate_Idea_GitHub_GitHubReady.ipynb
├── data/                                      
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data

This work uses Dubai real estate transaction data from **Dubai Land Department (DLD) Open Data**.

- Place your dataset in: `./data/`
- Update the filename in the notebook if yours differs.

---

## Quickstart

### 1) Create an environment and install dependencies

**Option A — venv**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda (often easier if GeoPandas causes issues)**
```bash
conda create -n dubai-real-estate python=3.11 -y
conda activate dubai-real-estate
pip install -r requirements.txt
```

### 2) Add the dataset
Put your CSV into `./data/`, for example:
```text
data/transactions-2025-11-20.csv
```

### 3) Run the notebook
```bash
jupyter lab
```
Open:
- `notebooks/Real_Estate_Idea_GitHub_GitHubReady.ipynb`

---

## Mispricing / “Arbitrage” explorer (optional)

The notebook includes an extra analysis that:
- splits the dataset into **90% modeling** and **10% arbitrage hold-out**
- trains the chosen model on the 90%
- predicts on the 10% and computes:
  - `MISPRICING = predicted - actual`
  - `MISPRICING_PCT = MISPRICING / actual`
- uses validation residual quantiles to form a simple prediction interval

### Mapping (optional)
If you run the mapping section:
- it geocodes `AREA_EN` via **Nominatim** (OpenStreetMap) and **caches** coordinates
- then visualizes points using GeoPandas + Contextily basemap

**Tip:** Geocoding is rate-limited; caching avoids repeat calls.

---



## Acknowledgements / Sources
- Dubai Land Department (DLD) Open Data (dataset provider)
- https://dubailand.gov.ae/en/open-data/real-estate-data#/

