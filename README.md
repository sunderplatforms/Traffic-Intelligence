# Smart Birmingham Traffic Intelligence
### Explainable Traffic Flow Prediction using Genetic Programming

## Project Overview

This project explores how **Computational Intelligence** can support **smart transportation** by modelling and predicting traffic flow into and out of Birmingham.

The core focus is **Genetic Programming (GP)**, applied via **Symbolic Regression**, to evolve a mathematical expression that predicts traffic flow directly from road traffic data. Unlike black-box ensemble models, symbolic regression produces a human-readable formula — in principle making the model's reasoning inspectable rather than opaque.

This is a personal continuation of a university dissertation project, rebuilt with a more rigorous evaluation methodology than the original coursework version (see **Key Findings** below for what changed and why).

Birmingham is the case study location because the Department for Transport publishes long-running, public road traffic count data for the city, covering **1993–2025** across **590 count points**, including vehicle class, directional flow, and raw hourly counts.

---

## Project Aim

Develop an explainable computational intelligence model that predicts traffic flow into and out of Birmingham using road traffic data and Genetic Programming, and rigorously benchmark it against conventional machine learning baselines.

---

## Project Objectives

1. Collect and prepare Birmingham road traffic count data.
2. Explore traffic patterns by road, year, hour, and direction of travel.
3. Build a Genetic Programming symbolic regression model to predict traffic flow.
4. Benchmark GP against Linear Regression, Random Forest, and Gradient Boosting.
5. Evaluate every model using MAE, RMSE, and R², under an evaluation scheme that doesn't let information leak between count points.
6. Extract and interpret the best evolved symbolic expression, including its complexity.
7. Identify which features actually drive predictions, correcting for known biases in feature importance methods.

---

## Background

Intelligent Transportation Systems treat traffic modelling and prediction as central to smart city infrastructure. Transport for West Midlands' Data Insight programme similarly emphasises data-driven forecasting and nowcasting for transport planning across the region.

This project builds a simplified but methodologically honest prototype of that kind of system, using Birmingham as the case study.

---

## Why Birmingham?

Birmingham is a major UK city with substantial, publicly documented traffic movement. DfT reports **3,738 million vehicle miles** travelled on Birmingham's roads in 2025 alone. The available datasets — count points, vehicle class breakdowns, annual average daily flow, directional flow, and raw counts — make it well suited to a directional traffic-flow prediction task.

---

## Computational Intelligence Approach

Genetic Programming is an evolutionary algorithm that performs symbolic regression:

1. Generate an initial population of random mathematical expressions.
2. Evaluate each expression's prediction error on the training data.
3. Select better-performing expressions.
4. Produce new expressions via crossover and mutation.
5. Repeat over many generations.
6. Return the best-evolved expression.

The appeal is explainability: instead of a black-box prediction, GP can in principle return something like *"traffic flow ≈ f(road type, hour, location)"* as an actual formula.

---

## Methodology / Pipeline

```
Raw DfT CSV (72,948 rows)
        ↓
Cleaning & type fixes (mixed-type columns, count_point_id → categorical)
        ↓
Feature engineering (is_morning_peak, is_evening_peak, is_peak_hour)
        ↓
Grouped train/test split (grouped by count_point_id — see Key Findings)
        ↓
Encoding:
   - Target encoding  → count_point_id, road_name   (high-cardinality)
   - One-hot encoding → road_type, direction_of_travel (low-cardinality)
        ↓
Baseline models: Linear Regression · Random Forest · Gradient Boosting
        ↓
Hyperparameter tuning (RandomizedSearchCV, grouped CV)
        ↓
Genetic Programming symbolic regression (gplearn)
        ↓
Evaluation: MAE · RMSE · R² (single split AND GroupKFold CV)
        ↓
Feature importance: impurity-based AND permutation-based
```

---

## Results

**GroupKFold cross-validated performance** (5 folds, grouped by `count_point_id` — this is the realistic "predicting at a new/unseen location" number):

| Model | RMSE | R² |
|---|---|---|
| Gradient Boosting | 205.7 | 0.924 |
| Linear Regression | 203.6 | 0.921 |
| Random Forest | 217.2 | 0.912 |

**Final holdout comparison** (tuned models + GP, single train/test split):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (tuned) | 74.9 | 148.7 | 0.964 |
| Gradient Boosting (tuned) | 88.7 | 181.0 | 0.947 |
| Genetic Programming | 98.9 | 206.4 | 0.931 |
| Linear Regression | 107.0 | 207.4 | 0.931 |

GP's evolved expression reached **length 75** (a deeply nested formula) — see `outputs/gp_expression.txt` for the full symbolic expression.

**Ablation — Random Forest without `count_point_id`:**

| Model variant | MAE | RMSE | R² |
|---|---|---|---|
| Full model (with location) | 74.9 | 148.7 | 0.964 |
| Location excluded | 107.5 | 203.8 | 0.933 |

See **Key Findings (4)** below for interpretation.

---

## Key Findings

These are the methodologically important findings from this project — including places where an initial naive approach gave misleading results, and what fixed them.

**1. A single train/test split overstates performance.**
An ungrouped split gave Random Forest an RMSE of ~117 and R² of 0.978. Grouping by `count_point_id` (so a location's data can't appear in both train and validation) revealed the realistic RMSE is closer to **205–220** — the original split let the model partly memorise each location's typical traffic rather than generalise.

**2. One-hot encoding a high-cardinality location ID breaks Linear Regression on unseen locations.**
Early versions one-hot encoded `count_point_id` (~600 dummy columns). Under grouped CV, every dummy for an unseen location is zero, causing Linear Regression to extrapolate wildly (RMSE ~3,600, R² ≈ −25.6). Switching to **target encoding** for `count_point_id` and `road_name` fixed this and cut the feature space from 623 columns down to 13.

**3. Genetic Programming is competitive but not obviously more interpretable.**
GP matched Linear Regression's accuracy (R² 0.931) — a genuinely good result. But its winning expression had a **program length of 75**, i.e. a deeply nested formula, not the simple, human-readable equation symbolic regression is often pitched as producing. This is a fair, reportable limitation rather than a failure.

**4. Feature importance needs a bias check — and an ablation confirms something more interesting than it first looked.**
Random Forest's built-in (impurity-based) feature importance is known to be biased toward high-cardinality features. Once `count_point_id` was target-encoded into a single column, it dominated both impurity-based (94.9%) and permutation-based importance, appearing to confirm that exact monitoring location is by far the single biggest driver of traffic volume.

A follow-up ablation — refitting Random Forest with `count_point_id` removed entirely — tells a more precise story. Accuracy only drops from **R² 0.964 (full model) to R² 0.933 (location excluded)**, a gap of just 0.031. In other words, most of what `count_point_id` appeared to be contributing was recoverable from `road_name` and `latitude`/`longitude` alone — once location is removed, **`road_name` becomes the dominant feature (81.7% impurity importance)**, followed by geographic coordinates.

This refines the original conclusion: it isn't granular per-count-point detail that drives traffic flow, but road identity (e.g. the M6 vs a residential side road) and broad geographic position. `count_point_id` was mostly acting as a fine-grained proxy for information already present in `road_name` and coordinates, not contributing large amounts of genuinely new signal.

**5. A parsimony-constrained Genetic Programming run found a genuinely interpretable formula, with minimal accuracy cost.**
The unconstrained GP run (as reported above) produced a 279-node expression achieving R² 0.931 — accurate, but no more readable than a black box. Sweeping `parsimony_coefficient` (which penalises program length during evolution) across `[0.0, 0.001, 0.005, 0.01, 0.05]` found a dramatically shorter alternative at `parsimony_coefficient=0.05`: a **5-node expression achieving R² 0.926** — a drop of only 0.6 percentage points in R² for a **98% reduction in complexity** (279 nodes → 5).

That expression, translated into feature names, is:

```
predicted_traffic ≈ location_average − (is_major_road × latitude)
```

Since Birmingham's latitude is ~52.5, the second term is effectively a near-constant downward adjustment applied only to major roads. In plain terms, GP rediscovered a simple, explainable rule: *predict each location's historical average traffic, with a fixed adjustment for major roads* — a genuinely human-readable model, not a black box. This is the strongest evidence in the project for the "explainable computational intelligence" aim stated at the outset.

One caveat: the length-vs-accuracy relationship was not perfectly monotonic across the sweep (`parsimony_coefficient=0.01` produced a *longer* expression, 173 nodes, than `0.001`'s 75 nodes, with worse RMSE too) — a hint that single runs at each setting might not be representative. A follow-up multi-seed robustness check confirmed this suspicion, with an important result of its own:

**Multi-seed robustness check (5 random seeds each):**

| Setting | Length (mean ± std, range) | R² (mean ± std) |
|---|---|---|
| Unconstrained (pc=0.0) | 165 ± 132 (range: 53–335) | 0.9306 ± 0.0025 |
| Parsimony-constrained (pc=0.05) | 6.6 ± 3.6 (range: 5–13) | 0.9266 ± 0.0011 |

Accuracy is stable across seeds either way (R² varies by <0.003). But **complexity is wildly seed-dependent without parsimony pressure** — the unconstrained run swings from 53 to 335 nodes purely based on random seed, meaning the original 279-node expression wasn't a meaningful, reproducible result — it was one arbitrary point in a huge range. With `parsimony_coefficient=0.05`, complexity reliably stays small (5–13 nodes) across every seed tested, at a cost of only 0.004 mean R².

**This is the project's strongest methodological finding**: symbolic regression's much-advertised interpretability is not automatic. Left unconstrained, GP's output complexity is essentially arbitrary and unreproducible even though its accuracy is stable. Interpretability had to be actively enforced via parsimony pressure — and once enforced, it was achieved reliably, at a small and consistent accuracy cost.

```
smart-birmingham-traffic-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── dft_rawcount_local_authority_id_141.csv   # not committed — see Setup
├── src/
│   └── traffic_flow_prediction.py                 # full pipeline (current: v3)
├── outputs/
│   ├── cv_results_baseline.csv
│   ├── holdout_results_tuned.csv
│   ├── gp_expression.txt
│   ├── feature_importance_impurity.csv
│   ├── feature_importance_permutation.csv
│   └── *.png                                       # comparison & importance charts
└── notebooks/
    └── exploration.ipynb                           # optional EDA, kept separate from the pipeline script
```

---

## Setup

```bash
git clone <your-repo-url>
cd smart-birmingham-traffic-intelligence
pip install -r requirements.txt
```

Download the Birmingham raw traffic count CSV from the [DfT road traffic statistics site](https://roadtraffic.dft.gov.uk/local-authorities/141) and place it at `data/dft_rawcount_local_authority_id_141.csv`, then update `DATA_PATH` in `src/traffic_flow_prediction.py` if needed.

```bash
python3 src/traffic_flow_prediction.py
```

---

## Future Work

- Ablation: permutation importance with `count_point_id` excluded, to isolate within-location drivers.
- Try a simpler GP constraint (e.g. `parsimony_coefficient`) to push toward shorter, more interpretable expressions, trading a little accuracy for genuine readability.
- Extend the directional-flow dataset to explicitly model traffic *into* vs *out of* Birmingham, rather than treating direction as a flat categorical feature.
- Package the pipeline as a small CLI or Streamlit dashboard for interactive exploration.