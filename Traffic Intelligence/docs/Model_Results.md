Model Results

Overview

Four models — Linear Regression, Random Forest, Gradient Boosting, and Genetic Programming (symbolic regression) — were trained and evaluated on the Birmingham DfT raw traffic count dataset, predicting `all_motor_vehicles` from location, road, direction and temporal features.

⸻

A note on methodology (read this before the numbers)

An earlier version of this project evaluated with a single, ungrouped train/test split, which let a count point's data appear in both train and test. That gave a misleadingly optimistic Random Forest result (RMSE ≈ 117, R² ≈ 0.978). Every result below instead either (a) reports a proper holdout split combined with tuned hyperparameters, or (b) uses **GroupKFold cross-validation grouped by `count_point_id`**, so a location's data can never appear in both the training and validation fold — the realistic test of "how well does this generalise to a location the model has never seen." See README Key Finding 1 for the full story. Numbers below are sourced from `outputs/cv_results_baseline.csv` and `outputs/holdout_results_tuned.csv`.

⸻

GroupKFold Cross-Validated Results (5 folds, baseline/default hyperparameters)

| Model | RMSE (mean) | R² (mean) |
|---|---:|---:|
| Gradient Boosting | 199.7 | 0.928 |
| Linear Regression | 203.6 | 0.921 |
| Random Forest | 213.0 | 0.916 |

This is the headline "predicting at an unseen location" number. All three models land in a similar, much more modest range than the naive single-split result — the realistic ceiling for this feature set is roughly R² 0.92–0.93, not 0.98.

⸻

Final Holdout Comparison (tuned models + Genetic Programming, single train/test split)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest (tuned) | 57.6 | 127.7 | 0.974 |
| Gradient Boosting (tuned) | 87.0 | 175.8 | 0.950 |
| Linear Regression | 107.0 | 207.4 | 0.931 |
| Genetic Programming (parsimony=0.05) | — | 212.2 | 0.927 |
| Genetic Programming (unconstrained) | — | 214.4 | 0.926 |

A single holdout split (rather than 5-fold CV) naturally has lower variance and typically a somewhat better score than the cross-validated figure above, particularly for Random Forest — both numbers are legitimate, they answer slightly different questions ("how good is this one trained model on this one split" vs "how well does this modelling approach generalise on average"). Report both, not just the more flattering one.

⸻

Ablation — Random Forest without `count_point_id`

| Model variant | MAE | RMSE | R² |
|---|---:|---:|---:|
| Full model (with location) | 74.9 | 148.7 | 0.964 |
| Location excluded | 107.5 | 203.8 | 0.933 |

Removing the location identifier only costs 0.031 R² — most of what `count_point_id` seemed to contribute is recoverable from `road_name` and coordinates alone. See `Feature_Analysis.md` and README Key Finding 4.

⸻

Findings

**Random Forest** is the strongest model under both evaluation schemes, and by a wider margin in the single-holdout comparison than under grouped CV — a reminder that a model's headline number depends heavily on the evaluation protocol.

**Gradient Boosting** is a close second on the holdout split and actually edges out the others under grouped CV with default hyperparameters — ensemble tree methods handle this feature set's non-linear structure well.

**Linear Regression** is competitive once `count_point_id`/`road_name` are target-encoded instead of one-hot encoded (R² 0.931) — a large improvement over an early one-hot-encoded version (see README Key Finding 2, which found one-hot encoding broke Linear Regression's extrapolation to unseen locations, RMSE ≈ 3,600).

**Genetic Programming** matches Linear Regression's accuracy (R² ≈ 0.927–0.931) while producing an actual formula rather than a set of coefficients — the main GP-specific finding is about the readability of that formula, not its accuracy; see README Key Finding 3 and 5, and `outputs/gp_expression.txt` / `outputs/gp_parsimony_sweep.csv`.

⸻

Conclusion

Random Forest is the strongest black-box benchmark. Genetic Programming reaches comparable accuracy to Linear Regression, and — critically — a parsimony-constrained GP run reaches a **5-node, human-readable expression** at a cost of well under 1 percentage point of R² (README Key Finding 5), which is the strongest evidence for this project's "explainable computational intelligence" aim.
