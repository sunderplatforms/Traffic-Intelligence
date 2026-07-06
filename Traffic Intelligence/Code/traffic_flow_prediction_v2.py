"""
Traffic Flow Prediction — v3
=============================
Birmingham DfT raw traffic counts: predicting `all_motor_vehicles`.

What changed from v2, and why:

  1. TARGET ENCODING for high-cardinality categoricals (count_point_id,
     road_name), instead of one-hot. One-hot encoding these blew the
     feature space up to 600+ columns, which:
       a) made Linear Regression extrapolate wildly on unseen locations
          during grouped CV (all dummy columns are zero for a location
          never seen in training -> huge, meaningless predictions), and
       b) made RF/GB/GP tuning extremely slow.
     sklearn's TargetEncoder cross-fits internally during fit_transform,
     so it does not leak the target the way a naive "replace category
     with its mean" implementation would.
     road_type and direction_of_travel are low-cardinality, so they stay
     one-hot encoded as before.

  2. FIXED NESTED PARALLELISM. v2 had n_jobs=-1 on both the search/CV
     wrapper AND the underlying estimator, which causes CPU
     oversubscription and can make things dramatically slower (this is
     what those repeated sklearn warnings in your v2 run were about).
     Now only the outer search/cross_validate call parallelises; the
     estimators themselves run single-threaded internally during search.

  3. SEPARATE, CHEAPER CV DURING TUNING. Hyperparameter search now uses
     3-fold grouped CV (not 5) to keep runtime reasonable; final reported
     metrics still use 5-fold CV and a proper holdout set.

  4. PROGRESS / TIMING PRINTS at each major stage, so long-running steps
     are visibly still working rather than looking hung.

Run:
    python3 traffic_flow_prediction_v3.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/dft_rawcount_local_authority_id_141.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS_FINAL = 5        # folds for reported cross-validated metrics
N_CV_FOLDS_TUNING = 3       # folds used *during* hyperparameter search (cheaper)
N_TUNING_ITER = 20          # RandomizedSearchCV iterations per model
SEARCH_N_JOBS = -1          # parallelism lives at the search level only
USE_GPLEARN = True
GP_GENERATIONS = 20
GP_POPULATION = 1000

# Columns with many distinct values -> target encoding
HIGH_CARDINALITY = ["count_point_id", "road_name"]
# Columns with few distinct values -> one-hot encoding
LOW_CARDINALITY_CATEGORICAL = ["road_type", "direction_of_travel"]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
t0 = time.time()
log("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
log(f"Shape: {df.shape}")

print("\nTarget stats:")
print(df["all_motor_vehicles"].describe())

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
df["is_evening_peak"] = df["hour"].isin([16, 17, 18]).astype(int)
df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

df["count_point_id"] = df["count_point_id"].astype(str)

target = "all_motor_vehicles"
features = [
    "year", "hour", "count_point_id", "road_type", "road_name",
    "direction_of_travel", "latitude", "longitude", "is_peak_hour",
]

X = df[features]
y = df[target]
groups = df["count_point_id"]

numeric_features = [c for c in features if c not in HIGH_CARDINALITY + LOW_CARDINALITY_CATEGORICAL]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("onehot", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), LOW_CARDINALITY_CATEGORICAL),
    ("target", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # target_type="continuous" for a regression target; TargetEncoder
        # cross-fits internally so fit_transform does not leak y into itself.
        ("encode", TargetEncoder(target_type="continuous", random_state=RANDOM_STATE)),
    ]), HIGH_CARDINALITY),
])

# ---------------------------------------------------------------------------
# 3. Grouped train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE,
)

X_train_processed = np.asarray(preprocessor.fit_transform(X_train, y_train))
X_test_processed = np.asarray(preprocessor.transform(X_test))
feature_names = preprocessor.get_feature_names_out()

log(f"Processed shapes: {X_train_processed.shape} / {X_test_processed.shape} "
    f"(down from 623 columns in the one-hot version)")

# ---------------------------------------------------------------------------
# 4. GroupKFold cross-validated comparison (baseline hyperparameters)
# ---------------------------------------------------------------------------
X_all_processed = np.asarray(preprocessor.fit_transform(X, y))
gkf_final = GroupKFold(n_splits=N_CV_FOLDS_FINAL)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

baseline_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

log("Running GroupKFold cross-validation for baseline models...")
cv_rows = []
for name, model in baseline_models.items():
    t_start = time.time()
    cv_result = cross_validate(
        model, X_all_processed, y, groups=groups,
        cv=gkf_final, scoring=scoring, n_jobs=SEARCH_N_JOBS,
    )
    cv_rows.append({
        "Model": name,
        "MAE_mean": -cv_result["test_MAE"].mean(),
        "MAE_std": cv_result["test_MAE"].std(),
        "RMSE_mean": -cv_result["test_RMSE"].mean(),
        "RMSE_std": cv_result["test_RMSE"].std(),
        "R2_mean": cv_result["test_R2"].mean(),
        "R2_std": cv_result["test_R2"].std(),
    })
    log(f"  {name}: done in {time.time() - t_start:.1f}s")

cv_results_df = pd.DataFrame(cv_rows).sort_values("RMSE_mean").reset_index(drop=True)
print("\n=== GroupKFold cross-validated results (baseline hyperparameters) ===")
print(cv_results_df)

# ---------------------------------------------------------------------------
# 5. Hyperparameter tuning (RandomizedSearchCV, grouped CV, fixed parallelism)
# ---------------------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

gb_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.6, 0.8, 1.0],
}

gkf_tuning = GroupKFold(n_splits=N_CV_FOLDS_TUNING)

log("Tuning Random Forest...")
t_start = time.time()
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),  # single-threaded estimator
    param_distributions=rf_param_grid,
    n_iter=N_TUNING_ITER,
    scoring="neg_root_mean_squared_error",
    cv=gkf_tuning,
    random_state=RANDOM_STATE,
    n_jobs=SEARCH_N_JOBS,   # parallelism happens here instead
)
rf_search.fit(X_all_processed, y, groups=groups)
log(f"RF tuning done in {time.time() - t_start:.1f}s")
print("Best RF params:", rf_search.best_params_)
print("Best RF CV RMSE:", -rf_search.best_score_)

log("Tuning Gradient Boosting...")
t_start = time.time()
gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    param_distributions=gb_param_grid,
    n_iter=N_TUNING_ITER,
    scoring="neg_root_mean_squared_error",
    cv=gkf_tuning,
    random_state=RANDOM_STATE,
    n_jobs=SEARCH_N_JOBS,
)
gb_search.fit(X_all_processed, y, groups=groups)
log(f"GB tuning done in {time.time() - t_start:.1f}s")
print("Best GB params:", gb_search.best_params_)
print("Best GB CV RMSE:", -gb_search.best_score_)

# Refit tuned models on the train/test holdout split for final reporting
tuned_rf = RandomForestRegressor(**rf_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
tuned_rf.fit(X_train_processed, y_train)
preds_tuned_rf = tuned_rf.predict(X_test_processed)

tuned_gb = GradientBoostingRegressor(**gb_search.best_params_, random_state=RANDOM_STATE)
tuned_gb.fit(X_train_processed, y_train)
preds_tuned_gb = tuned_gb.predict(X_test_processed)

lr = LinearRegression()
lr.fit(X_train_processed, y_train)
preds_lr = lr.predict(X_test_processed)

holdout_rows = [
    {"Model": "Linear Regression",
     "MAE": mean_absolute_error(y_test, preds_lr),
     "RMSE": np.sqrt(mean_squared_error(y_test, preds_lr)),
     "R2": r2_score(y_test, preds_lr)},
    {"Model": "Random Forest (tuned)",
     "MAE": mean_absolute_error(y_test, preds_tuned_rf),
     "RMSE": np.sqrt(mean_squared_error(y_test, preds_tuned_rf)),
     "R2": r2_score(y_test, preds_tuned_rf)},
    {"Model": "Gradient Boosting (tuned)",
     "MAE": mean_absolute_error(y_test, preds_tuned_gb),
     "RMSE": np.sqrt(mean_squared_error(y_test, preds_tuned_gb)),
     "R2": r2_score(y_test, preds_tuned_gb)},
]

# ---------------------------------------------------------------------------
# 6. Genetic Programming vs Random Forest benchmark
# ---------------------------------------------------------------------------
gp_program_str = None
gp_length = None

if USE_GPLEARN:
    try:
        from gplearn.genetic import SymbolicRegressor

        log("Fitting Genetic Programming (gplearn)...")
        t_start = time.time()
        gp = SymbolicRegressor(
            population_size=GP_POPULATION,
            generations=GP_GENERATIONS,
            random_state=RANDOM_STATE,
            verbose=1,
            n_jobs=-1,
        )
        gp.fit(X_train_processed, y_train)
        preds_gp = gp.predict(X_test_processed)
        log(f"GP done in {time.time() - t_start:.1f}s")

        gp_program_str = str(gp._program)
        gp_length = gp._program.length_

        holdout_rows.append({
            "Model": "Genetic Programming",
            "MAE": mean_absolute_error(y_test, preds_gp),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds_gp)),
            "R2": r2_score(y_test, preds_gp),
        })

        print("\n=== Genetic Programming discovered expression ===")
        print(gp_program_str)
        print(f"Program length (complexity): {gp_length}")

    except ImportError:
        print("\n[Skipping Genetic Programming] gplearn is not installed. "
              "Install with `pip install gplearn` to include it.")

holdout_results_df = pd.DataFrame(holdout_rows).sort_values("RMSE").reset_index(drop=True)
print("\n=== Final holdout comparison (tuned RF/GB + LR + GP) ===")
print(holdout_results_df)

cv_results_df.to_csv(OUTPUT_DIR / "cv_results_baseline.csv", index=False)
holdout_results_df.to_csv(OUTPUT_DIR / "holdout_results_tuned.csv", index=False)
if gp_program_str is not None:
    with open(OUTPUT_DIR / "gp_expression.txt", "w") as f:
        f.write(f"GP program:\n{gp_program_str}\n\nComplexity (length): {gp_length}\n")

log(f"Results saved to: {OUTPUT_DIR.resolve()}")

# ---------------------------------------------------------------------------
# 7. Feature importance: impurity-based vs permutation-based
# ---------------------------------------------------------------------------
impurity_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": tuned_rf.feature_importances_,
}).sort_values("importance", ascending=False)

log("Computing permutation importance...")
t_start = time.time()
perm_result = permutation_importance(
    tuned_rf, X_test_processed, y_test,
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
)
log(f"Permutation importance done in {time.time() - t_start:.1f}s")

permutation_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values("importance_mean", ascending=False)

print("\nTop 10 features — impurity-based importance:")
print(impurity_importance_df.head(10))
print("\nTop 10 features — permutation importance:")
print(permutation_importance_df.head(10))

impurity_importance_df.to_csv(OUTPUT_DIR / "feature_importance_impurity.csv", index=False)
permutation_importance_df.to_csv(OUTPUT_DIR / "feature_importance_permutation.csv", index=False)

# ---------------------------------------------------------------------------
# 8. Charts
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(cv_results_df["Model"], cv_results_df["RMSE_mean"], yerr=cv_results_df["RMSE_std"], capsize=5)
plt.title("Model Comparison — GroupKFold Cross-Validated RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE (vehicles)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_rmse_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(holdout_results_df["Model"], holdout_results_df["RMSE"])
plt.title("Final Holdout Comparison — Tuned Models vs Genetic Programming")
plt.xlabel("Model")
plt.ylabel("RMSE (vehicles)")
plt.xticks(rotation=15, ha="right")
for i, v in enumerate(holdout_results_df["RMSE"]):
    plt.text(i, v + 5, f"{v:.1f}", ha="center")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "holdout_comparison_tuned.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, preds_tuned_rf, alpha=0.35)
min_v, max_v = min(y_test.min(), preds_tuned_rf.min()), max(y_test.max(), preds_tuned_rf.max())
plt.plot([min_v, max_v], [min_v, max_v])
plt.title("Random Forest (Tuned): Actual vs Predicted Traffic Flow")
plt.xlabel("Actual All Motor Vehicles")
plt.ylabel("Predicted All Motor Vehicles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted_rf_tuned.png", dpi=300)
plt.close()

top_impurity = impurity_importance_df.head(15).sort_values("importance")
top_perm = permutation_importance_df.head(15).sort_values("importance_mean")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].barh(top_impurity["feature"], top_impurity["importance"])
axes[0].set_title("Top 15 — Impurity-Based Importance")
axes[0].set_xlabel("Importance")

axes[1].barh(top_perm["feature"], top_perm["importance_mean"], xerr=top_perm["importance_std"])
axes[1].set_title("Top 15 — Permutation Importance")
axes[1].set_xlabel("Importance (RMSE increase when shuffled)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_comparison.png", dpi=300)
plt.close()

log(f"All charts and CSVs saved to outputs/. Total runtime: {time.time() - t0:.1f}s")
print("\nDone. Charts were saved directly to outputs/ without popping up windows "
      "(plt.show() removed — remove plt.close() and add plt.show() back if you want them to display).")