# SAT-SA — Supervisory Analytics Tool for SOC Assessment

A prototype auditing tool that evaluates the quality of a Security Operations
Center's (SOC) incident-handling performance — built as a personal/portfolio
project based on **SIH 2026 problem statement SIH26157 (NTRO)**. Not an
official SIH submission; this is a solo-scoped exploration of the problem.

## What it does

Most SOC tools monitor *threats*. SAT-SA monitors the **analysts and
processes** that are supposed to be handling those threats — flagging
rushed closures, skipped escalations, boilerplate investigation notes,
missing telemetry on critical assets, and entities whose overall behavior
is a statistical outlier compared to their peers.

It runs on synthetic SOC data (alerts, cases, entities, assets) with
deliberately injected "bad behavior" as ground truth, so detection accuracy
can be measured with real Precision / Recall / F1 — not just demoed.

## Detection engine

**Layer A — Rule-based (record-level)**
| Rule | What it catches |
|---|---|
| `fast_closure_no_investigation` | High/Critical alerts closed in <3 min with <5 words of notes |
| `critical_no_escalation` | Critical alerts closed without Tier 2/IR escalation |
| `templated_investigation` | Boilerplate/copy-pasted investigator notes (TF-IDF + cosine similarity) |
| `missing_telemetry` | Critical assets with zero alert coverage |
| `stale_open_case` | Cases open >30 days |
| `root_cause_missing_on_critical` | Critical incidents closed with no documented root cause |

**Layer B — Statistical (entity-level)**
- Peer-benchmark Z-score outliers on Mean Time to Close (MTTC) and
  escalation rate, using **leave-one-out** Z-scores (so one severe outlier
  can't inflate the population variance and mask its own detection)
- IsolationForest for multivariate behavioral anomalies across entities

Every finding carries a `severity_score` (1–10), a `confidence` score, raw
`evidence`, and a plain-English executive summary generated per rule.

## Results (on freshly generated synthetic data)

```
Recall:    100.00%
Precision:  ~80%
F1 Score:   ~88%
```
Ground truth is generated alongside the synthetic data itself — every
injected anomaly is logged, so these numbers are measured, not estimated.
66 unit/integration tests (pytest) cover every detection rule and Flask route.

## Dashboard

A Flask web app with:
- **Command Center** — executive risk ranking, sector risk allocation, Manager/Analyst view toggle
- **Entity detail** — per-entity findings, time-series trend (MTTC / escalation rate / finding count over time)
- **Finding detail + trace view** — full alert → case → notes → rule → finding chain
- **Sector Benchmarks**, **Raw Anomalies Log**, **Engine Diagnostics**
- CSV and PDF export per entity

## Tech stack

- **Backend**: Python, Flask, pandas, scikit-learn (IsolationForest, TF-IDF), scipy
- **Reporting**: reportlab (PDF), Chart.js (dashboard charts)
- **Config**: all thresholds/weights centralized in `config.yaml`
- **Testing**: pytest

## Running it locally

```bash
python -m venv venv
# Windows: venv\Scripts\Activate.ps1   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python generate_dataset.py    # synthetic data + injected ground truth
python detection_engine.py    # run detection rules → data/findings.json
python evaluate_engine.py     # Precision / Recall / F1 against ground truth

python -m pytest -q           # run the test suite

python app.py                 # dashboard at http://127.0.0.1:5000
```

## Path to real data

This prototype runs entirely on synthetic data. A real deployment would
ingest from a SIEM/ticketing system (e.g. Splunk, ServiceNow, Jira Service
Management) — field mapping from the synthetic schema to typical SIEM/ITSM
fields is straightforward since the schema (`entity_id`, `alert_id`,
`case_id`, timestamps, severity, escalation, notes) mirrors common
ticketing data models.

## Disclaimer

Synthetic data only — no real SOC, organization, or individual's data is
used or represented anywhere in this project.
