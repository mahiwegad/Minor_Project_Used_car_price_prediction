# Used Car Price Prediction (India)

An end-to-end machine learning project that predicts the resale price of a used car in India, with a live Streamlit web app.

**Live app:** _paste your Streamlit link here_

## Problem statement
Buyers and sellers of used cars need a fair price. This project trains regression models on past CarDekho listings to estimate a car's selling price (in INR) from its brand, age, kilometres driven, fuel, transmission, engine and power.

## Dataset
- **Source:** [Vehicle dataset from CarDekho (Kaggle)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), file `Car details v3.csv`
- **Size:** _fill in rows x columns after loading (notebook prints it)_
- **Target:** `selling_price` (INR)
- **Features:** brand, car age, km driven, fuel, seller type, transmission, owner, mileage, engine, max power, seats

## Approach
1. **Cleaning:** removed duplicates, converted text columns with units (e.g. "19.7 kmpl") to numbers, handled missing values, removed outliers.
2. **Feature engineering:** `year` -> `car_age`, `name` -> `brand`.
3. **EDA:** price distribution, price vs age/km/power, price by fuel/transmission/brand, correlation heatmap.
4. **Preprocessing:** median/mode imputation, scaling, one-hot encoding inside a scikit-learn `Pipeline` (no data leakage). Target trained on `log(price)` because prices are right-skewed.
5. **Models compared:** Linear Regression, Random Forest, Gradient Boosting, XGBoost (same split, same pipeline, 5-fold CV).
6. **Tuning:** `RandomizedSearchCV` on the best tree model, with before/after comparison.
7. **Explainability:** permutation feature importance.
8. **Deployment:** Streamlit app.

## Results
_Replace with your real numbers from `model_comparison_results.csv`_

| Model | MAE (Rs) | RMSE (Rs) | R2 |
|---|---|---|---|
| Linear Regression | | | |
| Random Forest | | | |
| Gradient Boosting | | | |
| XGBoost | | | |
| **Best, tuned** | | | |

Charts are in the `images/` folder.

## Run it yourself
```bash
git clone <your-repo-url>
cd used-car-price-predictor
pip install -r requirements.txt
# put "Car details v3.csv" in this folder, then run all cells of used_car_price_prediction.ipynb
streamlit run app.py
```

## Project structure
```
used_car_price_prediction.ipynb   # full analysis and modelling
app.py                            # Streamlit app
car_price_model.joblib            # trained pipeline (created by the notebook)
model_meta.json                   # dropdown options and scores for the app
requirements.txt
images/                           # saved charts
```

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

## Future scope
Add city and car condition, try stacking ensembles, show prediction intervals, retrain on fresher listings.
