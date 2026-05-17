# Toy Dependency Learning for Calibration DAGs

This folder contains a small, self-contained toy simulation for a possible
extension: learn/refine calibration dependency edges from repeated
`check_data`/`diagnose`-like observations.

The goal is not to model hardware faithfully. The goal is to test whether a
simple online update rule can improve a noisy weighted prior over dependency
edges when the hidden dependency graph is known only to the simulator. This is a
toy model of a diagnostic layer, not an Optimus integration.

Vanilla Optimus `diagnose` is designed to repair state efficiently, not learn
graph topology. The learning examples therefore assume an extension or override
that records additional inconsistency traces.

## Edge convention

`u -> v` means calibration routine `u` depends on calibration routine `v`.
To keep the toy graph acyclic, valid candidate edges satisfy `u > v`.

When node `v` drifts, miscalibration can propagate in the reverse graph
direction to routines that depend on `v`.

## Key Terms

- Edge weight: the learner's current confidence that a candidate dependency is physically justified.
- Edge-weight threshold: the cutoff used to prune low-weight candidate edges from the effective dependency graph.
- Recall: the fraction of hidden true dependency edges retained after thresholding.
- Precision: the fraction of retained candidate edges that are hidden true dependencies.
- Recall guardrail: the minimum acceptable recall used when choosing an edge-weight threshold; it is not itself a graph cutoff.
- Revalidation cost: the drifted routine plus all transitive dependents that would be rechecked/revalidated under the thresholded graph.

This metric is a proxy for validation work, not a hardware timing or pulse-level recalibration measurement.

## Files

- `dependency_learning_demo.ipynb` — main artifact; includes the brief report, implementation walkthrough, figures, interpretation, and limitations.
- `simulate_dependency_learning.py` — runnable Python script.
- `outputs/proposal_summary_figure.png` — main figure.
- `outputs/edge_precision_recall_curve.png` — edge precision/recall curve.
- `outputs/cost_recall_tradeoff.png` — cost/recall guardrail figure.
- `outputs/metrics.json` — metrics and edge-weight threshold curves.

## Workflow

1. Generate a hidden true DAG and a dense set of candidate dependency edges.
2. Create a noisy physics-informed prior over candidate edge weights. Known or physics-informed prior edges receive high initial weights, spatially local speculative edges receive moderate weights, and other candidate edges receive lower weights.
3. Simulate drift events, propagation, and noisy `check_data` observations.
4. Update candidate edge weights in log-odds space from synthetic evidence.
5. Sweep edge-weight thresholds and select one that minimizes revalidation cost while satisfying the recall guardrail.
6. Write metrics and figures into `outputs/`.

## Run

```bash
python simulate_dependency_learning.py
```

or open and run the notebook top to bottom.

Optional CLI arguments include `--n-nodes`, `--n-events`, `--seed`, `--p-true`,
`--obs-noise`, `--output-dir`, `--no-spatial-locality`, and
`--propagation-prob`.

## Interpretation

The simulation is not hardware-realistic. It tests only whether a simple update rule can identify hidden dependency edges from controlled synthetic observations. The cost panel must be read with the recall panel: low revalidation cost at a high edge-weight threshold can indicate missed needed revalidations if the graph deletes true dependencies.

In the proposal summary figure, panels (1)-(4) correspond to precision/recall,
weight separation, revalidation cost, and the recall guardrail used to choose
the selected edge-weight threshold.
