# Toy Dependency Learning for Calibration DAGs

This folder contains a small, self-contained toy simulation for a proposal idea:
learn/refine calibration dependency edges from repeated `check_data`/`diagnose`-like observations.

## Edge convention

`u -> v` means calibration routine `u` depends on calibration routine `v`.
To keep the toy graph acyclic, valid candidate edges satisfy `u > v`.

## Files

- `dependency_learning_demo.ipynb` — main artifact; includes the brief report, implementation walkthrough, figures, interpretation, and limitations.
- `simulate_dependency_learning.py` — runnable Python script.
- `outputs/proposal_summary_figure.png` — main figure.
- `outputs/edge_precision_recall_curve.png` — edge precision/recall curve.
- `outputs/cost_recall_tradeoff.png` — cost/recall guardrail figure.
- `outputs/metrics.json` — metrics and threshold curves.

## Run

```bash
python simulate_dependency_learning.py --output-dir outputs
```

or open and run the notebook top to bottom.

## Interpretation

The simulation is not hardware-realistic. It tests only whether a simple update rule can identify hidden dependency edges from controlled synthetic observations. The cost panel must be read with the recall panel: low cost at high threshold can indicate missed needed recalibrations if the graph deletes true dependencies.
