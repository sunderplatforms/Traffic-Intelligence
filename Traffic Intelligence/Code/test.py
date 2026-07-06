"""
Traffic Flow Prediction — Model Comparison
-------------------------------------------
Compares Linear Regression, Random Forest, Gradient Boosting, and
(optionally) Genetic Programming (gplearn) on DfT traffic count data.

Fixes applied vs. the original draft:
  - OUTPUT_DIR is now a real pathlib.Path, created up front.
  - importance_df is built from rf.feature_importances_ before it's plotted.
  - count_point_id is treated as categorical (an ID), not numeric.
  - Duplicate model-fitting cells removed.
  - Added GroupKFold cross-validation (grouped by count_point_id) so the
    same physical count location can't leak between train and test.
  - gplearn import is optional/guarded so the rest of the script still
    runs if it isn't installed.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/dft_rawcount_local_authority_id_141.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

USE_GPLEARN = True  # set False to skip the genetic programming model entirely

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print(df.info())
print(df.isnull().sum())

print("\nTarget stats:")
print(df["all_motor_vehicles"].describe())

print("\nMean traffic by hour (top 10):")
print(df.groupby("hour")["all_motor_vehicles"].mean().sort_values(ascending=False).head(10))

print("\nMean traffic by road type:")
print(df.groupby("road_type")["all_motor_vehicles"].mean())

print("\nMean traffic by direction of travel:")
print(df.groupby("direction_of_travel")["all_motor_vehicles"].mean())

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
df["is_evening_peak"] = df["hour"].isin([16, 17, 18]).astype(int)
df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

target = "all_motor_vehicles"

# count_point_id is an identifier, not a continuous quantity — treat it as
# categorical so models don't infer a spurious ordering/magnitude from it.
df["count_point_id"] = df["count_point_id"].astype(str)

features = [
    "year",
    "hour",
    "count_point_id",
    "road_type",
    "road_name",
    "direction_of_travel",
    "latitude",
    "longitude",
    "is_peak_hour",
]

X = df[features]
y = df[target]

# Keep the grouping key around for GroupKFold, aligned to X's index.
groups = df.loc[X.index, "count_point_id"]

X_train, X_test, y_train, y_test, groups_train, _ = train_test_split(
    X, y, groups,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_features),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.array(X_train_processed)
X_test_processed = np.array(X_test_processed)

print("\nProcessed shapes:", X_train_processed.shape, X_test_processed.shape)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train_processed, y_train)
preds_lr = lr.predict(X_test_processed)

rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_processed, y_train)
preds_rf = rf.predict(X_test_processed)

gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
gb.fit(X_train_processed, y_train)
preds_gb = gb.predict(X_test_processed)

model_names = ["Linear Regression", "Random Forest", "Gradient Boosting"]
all_preds = [preds_lr, preds_rf, preds_gb]

# Optional: Genetic Programming via gplearn (guarded import)
if USE_GPLEARN:
    try:
        from gplearn.genetic import SymbolicRegressor

        gp = SymbolicRegressor(
            population_size=1000,
            generations=20,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        gp.fit(X_train_processed, y_train)
        preds_gp = gp.predict(X_test_processed)

        model_names.append("Genetic Programming")
        all_preds.append(preds_gp)
    except ImportError:
        print("\n[Skipping Genetic Programming] gplearn is not installed. "
              "Install with `pip install gplearn` to include it.")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
results = pd.DataFrame({
    "Model": model_names,
    "MAE": [mean_absolute_error(y_test, p) for p in all_preds],
    "RMSE": [np.sqrt(mean_squared_error(y_test, p)) for p in all_preds],
    "R2": [r2_score(y_test, p) for p in all_preds],
}).sort_values("RMSE").reset_index(drop=True)

print("\nModel comparison (single train/test split):")
print(results)

# ---------------------------------------------------------------------------
# Optional: GroupKFold cross-validation sanity check
# ---------------------------------------------------------------------------
# A single split can be noisy, and repeated count_point_id values could
# otherwise leak between train/test. This grouped CV keeps each count point
# entirely in one fold, giving a more honest estimate of generalisation.
print("\nGroupKFold cross-validation (RF, 5 folds, grouped by count_point_id):")
X_all_processed = np.array(preprocessor.fit_transform(X))
gkf = GroupKFold(n_splits=5)
cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    X_all_processed, y,
    groups=groups,
    cv=gkf,
    scoring="neg_root_mean_squared_error",
)
print("Fold RMSEs:", -cv_scores)
print("Mean RMSE:", -cv_scores.mean(), " Std:", cv_scores.std())

# ---------------------------------------------------------------------------
# Chart 1: Model comparison bar chart (RMSE)
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["RMSE"])
plt.title("Traffic Flow Prediction Model Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE (vehicles)")

for index, value in enumerate(results["RMSE"]):
    plt.text(index, value + 10, f"{value:.1f}", ha="center")

plt.tight_layout()
chart_path = OUTPUT_DIR / "model_comparison_rmse.png"
plt.savefig(chart_path, dpi=300)
plt.show()
print(f"Saved to: {chart_path}")

# ---------------------------------------------------------------------------
# Chart 2: Random Forest — Actual vs Predicted
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(y_test, preds_rf, alpha=0.35)

min_value = min(y_test.min(), preds_rf.min())
max_value = max(y_test.max(), preds_rf.max())
plt.plot([min_value, max_value], [min_value, max_value])

plt.title("Random Forest: Actual vs Predicted Traffic Flow")
plt.xlabel("Actual All Motor Vehicles")
plt.ylabel("Predicted All Motor Vehicles")
plt.tight_layout()

chart_path = OUTPUT_DIR / "actual_vs_predicted_random_forest.png"
plt.savefig(chart_path, dpi=300)
plt.show()
print(f"Saved to: {chart_path}")

# ---------------------------------------------------------------------------
# Chart 3: Random Forest feature importance
# ---------------------------------------------------------------------------
feature_names = preprocessor.get_feature_names_out()
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)

top_features = importance_df.head(15).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

chart_path = OUTPUT_DIR / "random_forest_feature_importance.png"
plt.savefig(chart_path, dpi=300)
plt.show()
print(f"Saved to: {chart_path}")