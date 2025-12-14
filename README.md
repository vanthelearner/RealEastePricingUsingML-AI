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


## Acknowledgements / Sources
- Dubai Land Department (DLD) Open Data (dataset provider)
- https://dubailand.gov.ae/en/open-data/real-estate-data#/

