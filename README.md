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

---

## Extension: Junction-Level Traffic Prediction

The original project aim was framed around traffic "into and out of Birmingham" generally. This extension takes that idea further: predicting traffic on the links connected to **specific named junctions**, using the trained model rather than historical averages — closer to the eventual goal of a junction-level traffic simulation tool.

### Scope

Only **31.9% of rows** (23,268 of 72,948) have both `start_junction_road_name` and `end_junction_road_name` populated — DfT only records junction endpoints for classified roads (A-roads and motorways), not minor streets. Within that subset:

- **170 distinct links** (count points), forming **172 near-unique (road, start-junction, end-junction) combinations** — only 10 count points had ambiguous multiple link definitions.
- Covers **23 major named roads**.

This means the junction tool covers Birmingham's major road network specifically, not every street — a real and honest scope limitation, not a shortcoming to hide.

### Two data-quality fixes found while building the graph

1. **Inconsistent road naming fragmented a single road into two identities** — e.g. `A38M` and `A38(M)` (the Aston Expressway) appeared as separate `road_name` values, which would have split one continuous road into two disconnected graph identities. Fixed by canonicalizing road names (stripping punctuation/case before comparison) and merging variants.
2. **A pervasive, structural ambiguity in junction labels — not just the one-off "LA Boundary" case.** Birmingham's A4040 (the Middle Ring Road) is crossed by nearly every major radial route (A38, A34, A41, A452, etc.) at *different* points around the ring, but DfT's junction label just names the crossing road ("meets the A4040") without saying where. A systematic check (`check_generic_junctions.py`, which measures the geographic spread of count points sharing each junction label) found this pattern affected **24 different labels**, not one — including A38, A34, A41, A4040, A452, A453, A441, A47, A4540, and more, with spreads up to 24km across as many as 11 different roads. The fix generalizes the original "LA Boundary" patch: **any junction label matching a known road name in the dataset is treated as generic and qualified by the road using it as an endpoint** (e.g. `A4040 (via A34)`, `A4040 (via A38)`), rather than hardcoding one label. After the fix, the graph grew from 126 to **194 junction nodes** (169 links), with the top-traffic ranking now dominated by genuinely specific locations (`Park Circus B4132 Waterlinks Boulevard`, `B4217`, `A4040 Wheelwright Road`, `A456/A457 roundabout`) instead of ambiguous bare road names absorbing traffic from many unrelated physical points.

**Known residual limitation**: qualifying a label by the road referencing it only resolves ambiguity when *multiple different roads* share that label — it does not resolve the same road reusing a bare number at more than one distinct point. The M6's bare junction numbers ("5", "6") were relabeled for consistency (`5 (via M6)`, `6 (via M6)`), but the underlying case found by the systematic check — "6" spanning 7.17km even with only the M6 referencing it — remains genuinely unresolved. This is a small, low-priority edge case affecting one motorway's numbering, documented here rather than pursued further given diminishing returns. Notably, after all fixes, the network's top three nodes by total connected traffic are all M6 junctions — a sensible result, since the M6 is Birmingham's dominant motorway corridor.

### Two-stage build

- **`check_generic_junctions.py`** — the systematic diagnostic described above: computes the geographic spread of count points sharing each junction label and flags any label that's likely generic/ambiguous, rather than relying on spotting issues manually one at a time.
- **`junction_network.py`** — builds the graph (junctions as nodes, links as edges) and computes a baseline network using **historical average traffic** per link. Produces `outputs/junction_network.png`, `junction_link_summary.csv`, and `junction_totals.csv`.
- **`junction_predictor.py`** — extends this by predicting traffic with the trained Random Forest model for **any chosen year and hour**, instead of a historical average, then aggregates predictions across every link touching a chosen junction. Exposes a reusable `predict_junction_traffic(junction_name, year, hour, ...)` function.
- **`junction_cli.py`** — an interactive command-line front-end for the above: trains the model once, then lets you repeatedly type a junction name, year, and hour and see the predicted breakdown immediately, without editing any code. This is the recommended way to actually explore the tool.

### Honest limitations

- **Not a geographic map.** Junction node positions use a force-directed layout, not real coordinates — the data identifies junctions by name, not by their own lat/long.
- **No turning-movement split.** The tool predicts total volume on each link touching a junction; it cannot say what proportion of that traffic turns left/right/continues straight, since DfT's raw counts don't include turning-movement data. A true simulation with that level of detail would need additional geometry and signal-timing data this dataset doesn't provide.
- **`flow_direction` (inbound/outbound/lateral) is relative to Birmingham's centre, not the specific junction being queried** — an approximate directional signal, not a precise "traffic arriving from the north at this exact junction" figure.
- **The two tools' totals are not directly comparable.** `junction_network.py`'s baseline averages both directions of travel *together* into one number per link; `junction_predictor.py` predicts each direction separately and sums them, and represents one specific hour rather than an all-hours average. Report these as distinct metrics, not the same thing measured twice.

### Example result — a genuine model validation, not just a number

At the A41 junction, one link (count point 7927, the A4540) showed westbound/inbound traffic exceeding eastbound/outbound at 8am (675.9 vs 594.8), which **flipped by 5pm** (outbound 678.8 vs inbound 641.4) — the classic AM-commute-in / PM-commute-out pattern, emerging from the model itself rather than being hand-coded. This isn't universal across every link (e.g. the A4400 near A38 stayed outbound-dominant at both times, which is also plausible — not every road carries a symmetric commuter flow), but it's a meaningful qualitative sanity check that the model captures real directional time-of-day structure.

---

## Repository Structure

This reflects the actual current layout (Alex's working setup), rather than a from-scratch ideal structure — safer than a risky physical reorganization given several scripts have hardcoded `DATA_PATH` values.

```
FYP v.2/                                       <- run scripts from here; outputs/ lands here
├── outputs/                                    generated charts, CSVs, gp_expression.txt
└── Traffic Intelligence/
    ├── dft_rawcount_local_authority_id_141.csv
    └── Code/
        ├── traffic_flow_prediction.py          main pipeline (rename from _v2.py/_v3.py for clarity —
        │                                        see note below)
        ├── junction_network.py                 historical junction network (baseline)
        ├── junction_predictor.py               model-based junction traffic predictor
        ├── junction_cli.py                     interactive CLI front-end for the above
        ├── check_generic_junctions.py           systematic check for generic/ambiguous junction labels
        └── check_junctions.py                  one-off diagnostic (junction field coverage)
```

**One safe rename worth doing**, purely for clarity (this is just a filename, so it's low-risk):

```bash
cd "/Users/Alex/Documents/FYP v.2/Traffic Intelligence/Code"
mv traffic_flow_prediction_v2.py traffic_flow_prediction.py
```

(The file has accumulated several rounds of fixes over the course of this project — GroupKFold evaluation, target encoding, the feature-importance ablation, GP parsimony sweep, multi-seed robustness, and the `flow_direction` feature — so despite the `_v2` filename, it reflects the final, most rigorous version. Renaming it removes that confusion for anyone reading the repo later, including yourself.)

---

## Setup

```bash
pip3 install -r requirements.txt
```

The raw CSV should already be at `Traffic Intelligence/dft_rawcount_local_authority_id_141.csv` (downloaded from the [DfT road traffic statistics site](https://roadtraffic.dft.gov.uk/local-authorities/141)). If it moves, update `DATA_PATH` at the top of each script — each one currently hardcodes the same absolute path.

Run from the `FYP v.2` folder (so `outputs/` lands in the right place):

```bash
python3 "Traffic Intelligence/Code/traffic_flow_prediction.py"   # main pipeline
python3 "Traffic Intelligence/Code/junction_network.py"          # junction baseline network
python3 "Traffic Intelligence/Code/junction_predictor.py"        # junction traffic predictor
```

---

## Future Work

- ~~Ablation: permutation importance with `count_point_id` excluded~~ — done; see Key Findings (4).
- ~~Try a parsimony-constrained GP for a shorter, more interpretable expression~~ — done; see Key Findings (5).
- ~~Extend to directional in/out-of-Birmingham modelling~~ — done via `flow_direction`; see the feature engineering in `traffic_flow_prediction.py`.
- ~~Junction-level traffic prediction~~ — done; see the Extension section above.
- ~~A true turning-movement breakdown at junctions~~ — not currently feasible; DfT's raw counts don't include the turning-movement or signal-timing data this would require. Documented as a permanent scope limitation rather than a to-do.
- ~~Package the junction predictor as a small interactive tool~~ — done via `junction_cli.py`.
- Re-run the GP parsimony sweep and robustness check with more random seeds per setting, for a smoother, more statistically robust tradeoff curve.
- Resolve the remaining M6 junction-number edge case (the same road, "6", reusing a bare number at two points 7.17km apart) — low priority given its small traffic footprint relative to the fix already applied.
- Integrate `check_generic_junctions.py` as a validation step that runs automatically before `junction_network.py` builds the graph, rather than as a separate manual script — would catch this class of issue automatically on any future data update.