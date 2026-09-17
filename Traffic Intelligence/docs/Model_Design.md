* Model Design

# Prediction Problem

The objective is to predict the value of
all_motor_vehicles using Birmingham
Department for Transport traffic count data.

Problem Type:
Regression

Target Variable:
all_motor_vehicles

Prediction Horizon:
Observed hourly traffic counts

# Selected Features

| Feature | Reason | Encoding |
|----------|----------|----------|
| year | captures long term traffic trends | numeric |
| hour | captures time of day effects | numeric |
| direction_of_travel | captures directional flow | one-hot (low cardinality) |
| road_type | major vs minor roads | one-hot (low cardinality) |
| road_name | identifies road characteristics | target encoding (high cardinality) |
| count_point_id | identifies traffic count location | target encoding (high cardinality) |
| latitude | geographic information | numeric |
| longitude | geographic information | numeric |
| is_peak_hour | engineered feature (rush-hour indicator) | numeric |
| flow_direction | engineered feature — inbound/outbound/lateral relative to Birmingham's centre | one-hot (low cardinality) |

`count_point_id` and `road_name` are high-cardinality (~600 and ~50 distinct values). They are **target encoded**, not one-hot encoded — an early version one-hot encoded them, which produced 600+ columns and made Linear Regression extrapolate wildly on unseen locations under grouped cross-validation (every dummy column is zero for a location never seen in training). See README Key Finding 2.

# Model Architecture

```
Data
 ↓
Cleaning (mixed-type columns, count_point_id -> categorical)
 ↓
Feature Engineering (is_morning_peak, is_evening_peak, is_peak_hour, flow_direction)
 ↓
Grouped Train/Test Split (grouped by count_point_id — see below)
 ↓
Encoding (target encoding for high-cardinality, one-hot for low-cardinality)
 ↓
Baseline Models (Linear Regression, Random Forest, Gradient Boosting)
 ↓
Hyperparameter Tuning (RandomizedSearchCV, grouped CV)
 ↓
Genetic Programming (gplearn SymbolicRegressor, parsimony-coefficient sweep)
 ↓
Evaluation (single holdout split AND GroupKFold cross-validation)
```

## Why a grouped split?

Rows from the same `count_point_id` are highly correlated (many hourly observations at the same physical location). An ordinary random train/test split lets a location's data appear in both train and test, so the model can partly memorise each location's typical traffic rather than generalise — this produced a misleadingly high RMSE (~117, R² 0.978) in an early version of this project. Splitting and cross-validating **by group** (`GroupKFold`, grouped on `count_point_id`) gives an honest estimate of performance at a genuinely unseen location, which is closer to the real deployment scenario. See README Key Finding 1.

## Interpretability: parsimony-constrained Genetic Programming

A single, accuracy-only GP fit produces a long, effectively black-box expression (hundreds of nodes) despite symbolic regression's usual "human-readable formula" pitch. `parsimony_coefficient` (which penalises program length during evolution) is swept across several values to trade a small amount of accuracy for a much shorter, genuinely readable expression — and the result is checked across multiple random seeds, since GP's output complexity turned out to be highly seed-dependent when parsimony pressure isn't applied. See README Key Finding 5 for the result.

# Metrics

MAE, RMSE, R² — reported both as a single train/test holdout comparison and as 5-fold GroupKFold cross-validated means/stds.
