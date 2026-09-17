# Exploratory Data Analysis Findings

## Dataset Overview

| Metric | Value |
|----------|----------|
| Rows | 72,948 |
| Columns | 32 |
| Years Covered | 2000–2025 |
| Count Points | 562 |
| Unique Roads (raw `road_name` values) | 50 |
| Unique Roads (after canonicalizing name variants) | 49 |
| Target Variable | all_motor_vehicles |

The dataset contains Birmingham road traffic count observations collected by the Department for Transport. Each observation represents traffic recorded at a specific count point, direction and hour.

Two raw `road_name` values — `A38M` and `A38(M)` — refer to the same physical road (the Aston Expressway) under inconsistent spellings; the 49 canonical roads figure merges them, using the same `canonical_road_name()` cleaning applied in the junction network extension (see README → Extension: Junction-Level Traffic Prediction).

---

## Missing Data Analysis

The following fields contain substantial missing values:

| Column | Missing Values |
|----------|----------|
| start_junction_road_name | 49,680 |
| end_junction_road_name | 49,680 |
| link_length_km | 49,680 |
| link_length_miles | 49,680 |

These are concentrated on minor/unclassified roads — DfT only records junction endpoints for classified roads (A-roads and motorways). This is the subset used for the junction-level extension (see `docs/Model_Design.md` and the README).

---

## Traffic by Road Type

| Road Type | Mean Traffic |
|------------|------------:|
| Major | 1304.4 |
| Minor | 177.4 |

### Finding

Major roads carry roughly 7x more traffic than minor roads on average — road classification alone is a strong predictor of traffic flow.

---

## Traffic by Direction

| Direction | Mean Traffic |
|------------|------------:|
| West | 560.1 |
| East | 547.8 |
| North | 527.4 |
| South | 522.2 |

### Finding

Raw compass direction is fairly balanced and, on its own, a weak signal — this is the motivation for the engineered `flow_direction` feature (inbound/outbound/lateral relative to Birmingham's centre; see `Feature_Engineering.md`), which is a much more informative recombination of direction + location than direction alone.

---

## Traffic by Hour

Highest mean traffic occurs at:

| Hour | Mean Traffic |
|------|------------:|
| 08:00 | 623.4 |
| 16:00 | 617.3 |
| 17:00 | 614.5 |
| 15:00 | 567.9 |
| 07:00 | 564.4 |
| 18:00 | 528.0 |

### Finding

A clear morning/evening commuting pattern, supporting the `is_morning_peak` / `is_evening_peak` / `is_peak_hour` engineered features.

---

## Highest Traffic Roads (canonicalized)

| Road | Mean Traffic |
|--------|------------:|
| M6 | 4062.0 |
| A38M (merges `A38M` + `A38(M)`) | 3318.5 |
| A4400 | 1670.5 |
| A38 | 1527.0 |
| A45 | 1519.8 |
| A4540 | 1514.1 |
| A456 | 1272.5 |
| A34 | 1198.0 |
| A4041 | 1114.6 |
| A452 | 1094.9 |

### Finding

Motorways and major arterial routes dominate Birmingham traffic movement, with the M6 far ahead of every other road. This table canonicalizes the `A38M`/`A38(M)` spelling variants into one row — an earlier pass over this data (see `Findings/EDA_Report1.md`) reported them as two separate roads, which was itself one of the data-quality issues this project found and fixed.

---

## Summary

* Major roads carry substantially more traffic than minor roads.
* Traffic follows a clear morning/evening commuting pattern.
* A small number of roads (led by the M6) dominate Birmingham's traffic volume.
* Raw compass direction alone is a weak feature; combined with location it becomes much more informative (`flow_direction`).
* Road-name spelling is inconsistent in the raw data and needs canonicalizing before being used as an identity (road-level analysis, and the junction graph, both depend on this).

For model performance and feature importance — which depend on modelling decisions (grouped cross-validation, encoding choice) rather than being pure EDA — see `Model_Results.md` and `Feature_Analysis.md`, and the README's Results / Key Findings sections for the fully worked methodology.
