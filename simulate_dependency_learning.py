#!/usr/bin/env python3
"""Toy dependency-graph learning simulation for quantum calibration DAGs.

Edge convention used throughout:
    u -> v means "calibration routine u depends on calibration routine v".

To keep the graph acyclic, valid edges only point from a higher-index node to a
lower-index node. When node v drifts, badness propagates in the reverse edge
direction to dependents u.

This file is intentionally self-contained. It generates a hidden "true" DAG,
constructs a noisy weighted prior over candidate dependencies, simulates noisy
check_data/diagnose-style observations, updates edge weights, and writes the
figures/metrics used by the companion notebook.

The simulation is a toy diagnostic layer, not an implementation of vanilla
Optimus diagnose behavior. Vanilla diagnose is designed to repair state
efficiently; this demo instead uses diagnose-like inconsistency traces to update
confidence in candidate dependency edges.

Terminology:
    edge-weight threshold
        Cutoff applied to candidate edge weights to decide which edges remain in
        the effective dependency graph.
    recall guardrail
        Minimum acceptable dependency-edge recall used when selecting an
        edge-weight threshold. It is a model-selection constraint, not a graph
        pruning cutoff.
    revalidation cost
        Toy proxy for validation work: the drifted routine plus all transitive
        dependents that would be rechecked/revalidated under the thresholded
        graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

# Keep plotting libraries from trying to write caches under a locked-down home directory.
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "optimus_prime_xdg_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "optimus_prime_mplconfig"))

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

try:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    average_precision_score = None
    precision_recall_curve = None
    SKLEARN_AVAILABLE = False


Edge = tuple[int, int]


def load_pyplot():
    """Import Matplotlib lazily with a non-interactive backend.

    The script can be run in headless environments, so plotting uses Agg and
    avoids importing Matplotlib until an output figure is actually requested.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the toy learning experiment.

    The defaults are chosen to make the demo stable and interpretable rather
    than hardware-realistic. Fields are grouped by role: graph size and noise,
    toy propagation assumptions, prior construction, observation sampling, and
    log-odds update strengths.
    """

    n_nodes: int = 30
    n_events: int = 600
    seed: int = 7
    p_true: float = 0.08
    obs_noise: float = 0.05
    output_dir: Path = Path("outputs")

    # Toy physics knobs.
    spatial_locality: bool = True
    locality_scale: float = 0.35
    propagation_prob: float = 0.80

    # The noisy known/physics-informed prior graph contains real edges plus false
    # positives. It forms the high-confidence part of the candidate prior.
    known_true_fraction: float = 0.55
    known_false_positive_fraction: float = 0.05

    # Synthetic check_data/diagnose observation model.
    affected_observation_prob: float = 0.95
    background_observation_prob: float = 0.25
    diagnose_success_prob: float = 0.75

    # Log-odds update magnitudes. Positive diagnose evidence is intentionally
    # stronger than ambiguous co-failure evidence.
    eta_pos: float = 0.65
    eta_neg: float = 0.35
    eta_neg_mild: float = 0.08
    eta_coaffected: float = 0.03


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Map log-odds values back to probabilities in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    """Map probabilities to log-odds, clipping endpoints for numerical safety."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def valid_dag_edges(n_nodes: int) -> list[Edge]:
    """Return all candidate edges allowed by the fixed DAG ordering.

    Requiring u > v for each u -> v edge prevents cycles while preserving the
    dependent -> dependency convention used throughout the simulation.
    """
    return [(u, v) for u in range(n_nodes) for v in range(u)]


def make_grid_coordinates(n_nodes: int) -> np.ndarray:
    """Assign each calibration node a simple 2D location for toy spatial locality."""
    side = int(math.ceil(math.sqrt(n_nodes)))
    rows = int(math.ceil(n_nodes / side))
    coords = np.zeros((n_nodes, 2), dtype=float)
    for node in range(n_nodes):
        x = node % side
        y = node // side
        coords[node, 0] = x / max(side - 1, 1)
        coords[node, 1] = y / max(rows - 1, 1)
    return coords


def edge_distances(edges: list[Edge], coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between endpoints of candidate edges."""
    if not edges:
        return np.array([], dtype=float)
    return np.array([np.linalg.norm(coords[u] - coords[v]) for u, v in edges])


def generate_hidden_true_dag(
    config: SimulationConfig, rng: np.random.Generator
) -> tuple[nx.DiGraph, set[Edge], np.ndarray, list[Edge], np.ndarray]:
    """Generate the hidden true dependency DAG and candidate edge universe.

    The hidden DAG is the evaluation target and is not observed by the learning
    rule. When spatial locality is enabled, nearby node pairs are more likely to
    be true dependencies, giving the prior a useful but imperfect signal.
    """
    coords = make_grid_coordinates(config.n_nodes)
    edges = valid_dag_edges(config.n_nodes)
    distances = edge_distances(edges, coords)

    if edges and config.spatial_locality:
        affinity = np.exp(-distances / max(config.locality_scale, 1e-6))
        probabilities = config.p_true * affinity / max(float(np.mean(affinity)), 1e-12)
        probabilities = np.clip(probabilities, 0.0, min(0.60, max(config.p_true * 5.0, config.p_true)))
    else:
        probabilities = np.full(len(edges), config.p_true, dtype=float)

    selected = rng.random(len(edges)) < probabilities
    true_edges = {edge for edge, is_selected in zip(edges, selected) if is_selected}

    # Keep demos meaningful for very small nonzero p_true settings.
    if config.p_true > 0.0 and edges and not true_edges:
        true_edges.add(edges[int(rng.integers(0, len(edges)))])

    graph = nx.DiGraph()
    graph.add_nodes_from(range(config.n_nodes))
    graph.add_edges_from(true_edges)
    return graph, true_edges, coords, edges, distances


def choose_without_replacement(
    rng: np.random.Generator, items: list[Edge], n_items: int
) -> set[Edge]:
    """Sample up to n_items distinct edges from a list."""
    if not items or n_items <= 0:
        return set()
    n_items = min(n_items, len(items))
    indices = rng.choice(len(items), size=n_items, replace=False)
    return {items[int(i)] for i in np.atleast_1d(indices)}


def create_prior_weights(
    config: SimulationConfig,
    rng: np.random.Generator,
    edges: list[Edge],
    true_edges: set[Edge],
    distances: np.ndarray,
) -> tuple[np.ndarray, set[Edge], np.ndarray]:
    """Create the dense weighted candidate graph used as the learner's prior.

    Known or physics-informed prior edges receive high initial weights,
    spatially local speculative edges receive moderate weights, and other
    candidate edges receive lower weights. The prior intentionally contains both
    true edges and false positives so the learning rule has something meaningful
    to refine.
    """
    true_edge_list = [edge for edge in edges if edge in true_edges]
    false_edge_list = [edge for edge in edges if edge not in true_edges]

    known_true_count = int(round(config.known_true_fraction * len(true_edge_list)))
    known_false_count = int(round(config.known_false_positive_fraction * len(false_edge_list)))
    known_edges = choose_without_replacement(rng, true_edge_list, known_true_count)
    known_edges |= choose_without_replacement(rng, false_edge_list, known_false_count)

    local_cutoff = float(np.quantile(distances, 0.30)) if len(distances) else 0.0
    local_mask = distances <= local_cutoff
    weights = np.zeros(len(edges), dtype=float)

    for idx, edge in enumerate(edges):
        if edge in known_edges:
            # Existing/physics-informed Optimus-like dependencies get high prior weight.
            weights[idx] = rng.normal(0.82, 0.05)
        elif local_mask[idx]:
            # Spatially local speculative dependencies are plausible but uncertain.
            weights[idx] = rng.normal(0.43, 0.06)
        else:
            # Long-range speculative dependencies start with low prior confidence.
            weights[idx] = rng.normal(0.20, 0.04)

    return np.clip(weights, 0.02, 0.98), known_edges, local_mask


def simulate_hidden_event(
    true_graph: nx.DiGraph, config: SimulationConfig, rng: np.random.Generator
) -> tuple[int, set[int], dict[int, int]]:
    """Simulate one drift event and its propagation through true dependencies.

    A root-cause calibration node drifts. Because edges are dependent -> dependency,
    badness propagates to dependents by traversing predecessor links.
    """
    root = int(rng.integers(0, config.n_nodes))
    affected = {root}
    cause_parent: dict[int, int] = {}
    queue = [root]

    while queue:
        dependency = queue.pop(0)
        for dependent in true_graph.predecessors(dependency):
            if dependent in affected:
                continue
            if rng.random() < config.propagation_prob:
                affected.add(dependent)
                cause_parent[dependent] = dependency
                queue.append(dependent)

    return root, affected, cause_parent


def simulate_check_data_observations(
    config: SimulationConfig,
    affected: set[int],
    root: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy check_data states for a sampled subset of calibration nodes.

    This approximates check_data along sampled candidate paths: affected nodes
    are likely to be checked, and unrelated nodes are occasionally checked as
    controls. Edge updates only use candidate edges whose endpoints were checked.
    """
    affected_state = np.zeros(config.n_nodes, dtype=bool)
    affected_state[list(affected)] = True

    observe_prob = np.where(
        affected_state, config.affected_observation_prob, config.background_observation_prob
    )
    observed_mask = rng.random(config.n_nodes) < observe_prob
    observed_mask[root] = True

    noisy_state = affected_state.copy()
    noisy_state ^= rng.random(config.n_nodes) < config.obs_noise
    return observed_mask, noisy_state


def update_weights_from_observations(
    weights: np.ndarray,
    edges: list[Edge],
    observed_mask: np.ndarray,
    noisy_state: np.ndarray,
    cause_parent: dict[int, int],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Update candidate edge weights from synthetic diagnose/check_data evidence.

    Updates are performed in log-odds space so repeated evidence compounds
    smoothly while final weights remain probabilities. Positive diagnose
    evidence strengthens a direct candidate dependency; contradictory check_data
    evidence weakens it.
    """
    logits = logit(weights)

    for idx, (u, v) in enumerate(edges):
        if not (observed_mask[u] and observed_mask[v]):
            continue

        # diagnose evidence: the simulated diagnostic routine sometimes surfaces
        # the direct dependency that transmitted the failure to u.
        diagnosed_direct_cause = cause_parent.get(u) == v and rng.random() < config.diagnose_success_prob
        if diagnosed_direct_cause and noisy_state[u] and noisy_state[v]:
            logits[idx] += config.eta_pos
            continue

        # check_data evidence: contradictions weaken candidate dependencies.
        if noisy_state[u] and not noisy_state[v]:
            # u failed while candidate dependency v appears in spec.
            logits[idx] -= config.eta_neg
        elif noisy_state[v] and not noisy_state[u]:
            # v failed but u stayed in spec. Propagation is probabilistic, so this is weak.
            logits[idx] -= config.eta_neg_mild
        elif noisy_state[u] and noisy_state[v]:
            # Co-failure is only weak support because common/transitive causes are possible.
            logits[idx] += config.eta_coaffected

    return sigmoid(logits)


def precision_recall_f1(
    labels: np.ndarray, scores: np.ndarray, edge_weight_threshold: float
) -> dict[str, float]:
    """Compute edge precision, recall, and F1 at one edge-weight threshold."""
    pred = scores >= edge_weight_threshold
    positives = labels == 1
    tp = int(np.sum(pred & positives))
    fp = int(np.sum(pred & ~positives))
    fn = int(np.sum(~pred & positives))

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def manual_pr_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Approximate a precision/recall curve without scikit-learn."""
    edge_weight_thresholds = np.linspace(0.0, 1.0, 201)
    precision_values = []
    recall_values = []
    for edge_weight_threshold in edge_weight_thresholds:
        prf = precision_recall_f1(labels, scores, float(edge_weight_threshold))
        precision_values.append(prf["precision"])
        recall_values.append(prf["recall"])

    precision_arr = np.array(precision_values)
    recall_arr = np.array(recall_values)
    order = np.argsort(recall_arr)
    auc_pr = float(np.trapz(precision_arr[order], recall_arr[order]))
    return recall_arr[order], precision_arr[order], auc_pr


def pr_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return recall, precision, and AUC-PR for candidate edge scores."""
    if SKLEARN_AVAILABLE and len(np.unique(labels)) == 2:
        assert precision_recall_curve is not None
        assert average_precision_score is not None
        precision, recall, _ = precision_recall_curve(labels, scores)
        order = np.argsort(recall)
        auc_pr = float(average_precision_score(labels, scores))
        return recall[order], precision[order], auc_pr
    return manual_pr_curve(labels, scores)


def edge_recall_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Summarize how well edge scores rank and retain hidden true dependencies."""
    at_half = precision_recall_f1(labels, scores, 0.5)

    best = {"edge_weight_threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for edge_weight_threshold in np.linspace(0.0, 1.0, 201):
        prf = precision_recall_f1(labels, scores, float(edge_weight_threshold))
        if prf["f1"] > best["f1"]:
            best = {"edge_weight_threshold": float(edge_weight_threshold), **prf}

    _, _, auc_pr = pr_curve(labels, scores)
    return {
        "auc_pr": auc_pr,
        "precision_at_0_5": at_half["precision"],
        "recall_at_0_5": at_half["recall"],
        "f1_at_0_5": at_half["f1"],
        "best_edge_weight_threshold": best["edge_weight_threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
    }


def graph_from_scores(
    n_nodes: int, edges: list[Edge], scores: np.ndarray, edge_weight_threshold: float
) -> nx.DiGraph:
    """Graph formed by retaining candidate edges at or above an edge-weight threshold."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edge for edge, score in zip(edges, scores) if score >= edge_weight_threshold)
    return graph


def mean_revalidation_cost(graph: nx.DiGraph, roots: Iterable[int]) -> float:
    """Estimate routines revalidated after a drift under a dependency graph.

    For each drift root, revalidation cost is one for the root plus every
    transitive dependent reached by reversing the dependent -> dependency graph.
    """
    dependency_to_dependent = graph.reverse(copy=True)
    revalidation_costs = [
        1 + len(nx.descendants(dependency_to_dependent, int(root))) for root in roots
    ]
    return float(np.mean(revalidation_costs)) if revalidation_costs else 0.0


def cost_curve(
    config: SimulationConfig,
    true_graph: nx.DiGraph,
    edges: list[Edge],
    initial_weights: np.ndarray,
    learned_weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Evaluate revalidation cost across edge-weight thresholds.

    The same sampled drift roots are used for the prior, learned, and true graphs
    so their cost curves are directly comparable.
    """
    roots = rng.integers(0, config.n_nodes, size=max(250, config.n_events // 2))
    edge_weight_thresholds = np.linspace(0.0, 1.0, 51)
    prior_costs = []
    learned_costs = []
    for edge_weight_threshold in edge_weight_thresholds:
        prior_graph = graph_from_scores(
            config.n_nodes, edges, initial_weights, float(edge_weight_threshold)
        )
        learned_graph = graph_from_scores(
            config.n_nodes, edges, learned_weights, float(edge_weight_threshold)
        )
        prior_costs.append(mean_revalidation_cost(prior_graph, roots))
        learned_costs.append(mean_revalidation_cost(learned_graph, roots))

    true_cost = mean_revalidation_cost(true_graph, roots)
    return edge_weight_thresholds, np.array(prior_costs), np.array(learned_costs), true_cost, roots


def edge_weight_threshold_tradeoff_curve(
    config: SimulationConfig,
    edges: list[Edge],
    labels: np.ndarray,
    scores: np.ndarray,
    roots: Iterable[int],
    edge_weight_thresholds: np.ndarray,
) -> dict[str, list[float]]:
    """Revalidation cost and edge-quality metrics for each thresholded graph."""
    mean_revalidation_costs = []
    precision_values = []
    recall_values = []
    missed_dependency_rates = []
    f1_values = []

    for edge_weight_threshold in edge_weight_thresholds:
        threshold_float = float(edge_weight_threshold)
        graph = graph_from_scores(config.n_nodes, edges, scores, threshold_float)
        prf = precision_recall_f1(labels, scores, threshold_float)
        mean_revalidation_costs.append(mean_revalidation_cost(graph, roots))
        precision_values.append(prf["precision"])
        recall_values.append(prf["recall"])
        missed_dependency_rates.append(1.0 - prf["recall"])
        f1_values.append(prf["f1"])

    return {
        "edge_weight_thresholds": edge_weight_thresholds.astype(float).tolist(),
        "mean_revalidated_routines_per_drift": [
            float(value) for value in mean_revalidation_costs
        ],
        "edge_precision": [float(value) for value in precision_values],
        "edge_recall": [float(value) for value in recall_values],
        "missed_dependency_rate": [float(value) for value in missed_dependency_rates],
        "f1": [float(value) for value in f1_values],
    }


def select_edge_weight_threshold(
    tradeoff: dict[str, list[float]], recall_guardrail: float = 0.90
) -> dict[str, float | str]:
    """Pick the lowest-revalidation-cost threshold that satisfies the recall guardrail."""
    edge_weight_thresholds = np.array(tradeoff["edge_weight_thresholds"], dtype=float)
    costs = np.array(tradeoff["mean_revalidated_routines_per_drift"], dtype=float)
    precision = np.array(tradeoff["edge_precision"], dtype=float)
    recall = np.array(tradeoff["edge_recall"], dtype=float)
    f1 = np.array(tradeoff["f1"], dtype=float)

    eligible_indices = np.flatnonzero(recall >= recall_guardrail)
    if len(eligible_indices):
        eligible_costs = costs[eligible_indices]
        min_cost = float(np.min(eligible_costs))
        candidate_indices = eligible_indices[np.isclose(eligible_costs, min_cost)]
        best_index = int(candidate_indices[np.argmax(f1[candidate_indices])])
        selection_reason = (
            f"min revalidation cost subject to recall guardrail >= {recall_guardrail:.2f}"
        )
    else:
        best_index = int(np.argmax(f1))
        selection_reason = (
            f"fallback to best F1; no edge-weight threshold met "
            f"the recall guardrail >= {recall_guardrail:.2f}"
        )

    return {
        "selected_edge_weight_threshold": float(edge_weight_thresholds[best_index]),
        "recall_guardrail": float(recall_guardrail),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1[best_index]),
        "revalidation_cost": float(costs[best_index]),
        "missed_dependency_rate": float(1.0 - recall[best_index]),
        "selection_reason": selection_reason,
    }


def select_operating_threshold(
    tradeoff: dict[str, list[float]], recall_guardrail: float = 0.90
) -> dict[str, float | str]:
    """Compatibility wrapper for the old threshold-selection helper name.

    Prefer :func:`select_edge_weight_threshold` in new code. This wrapper keeps
    older notebook kernels/cells working while the project uses the clearer
    edge-weight-threshold terminology.
    """
    return select_edge_weight_threshold(tradeoff, recall_guardrail=recall_guardrail)


def shade_recall_region(
    ax: plt.Axes,
    edge_weight_thresholds: np.ndarray,
    recall_values: np.ndarray,
    recall_guardrail: float = 0.90,
) -> None:
    """Shade edge-weight thresholds whose learned recall satisfies the guardrail."""
    mask = recall_values >= recall_guardrail
    if not np.any(mask):
        return
    ax.axvspan(
        float(np.min(edge_weight_thresholds[mask])),
        float(np.max(edge_weight_thresholds[mask])),
        color="#2ca02c",
        alpha=0.10,
        label=f"Recall guardrail met (>= {recall_guardrail:.2f})",
    )


def plot_pr_curves(
    labels: np.ndarray, initial_weights: np.ndarray, learned_weights: np.ndarray, output_path: Path
) -> None:
    """Write the standalone edge precision/recall curve figure."""
    plt = load_pyplot()
    initial_recall, initial_precision, initial_auc = pr_curve(labels, initial_weights)
    learned_recall, learned_precision, learned_auc = pr_curve(labels, learned_weights)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(initial_recall, initial_precision, label=f"Initial prior (AUC-PR={initial_auc:.3f})")
    ax.plot(learned_recall, learned_precision, label=f"Learned weights (AUC-PR={learned_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Dependency Edge Precision-Recall")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def weight_matrix(n_nodes: int, edges: list[Edge], scores: np.ndarray) -> np.ndarray:
    """Convert sparse edge scores into a node-by-node matrix for heatmaps."""
    matrix = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    for (u, v), score in zip(edges, scores):
        matrix[u, v] = score
    return matrix


def plot_weight_heatmaps(
    n_nodes: int,
    edges: list[Edge],
    true_edges: set[Edge],
    initial_weights: np.ndarray,
    learned_weights: np.ndarray,
    output_path: Path,
) -> None:
    """Write before/after heatmaps of candidate dependency weights."""
    plt = load_pyplot()
    matrices = [
        ("Before: physics-informed prior", weight_matrix(n_nodes, edges, initial_weights)),
        ("After: learned from observations", weight_matrix(n_nodes, edges, learned_weights)),
    ]

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#ececec")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharex=True, sharey=True)

    image = None
    for ax, (title, matrix) in zip(axes, matrices):
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0.0, vmax=1.0, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Dependency node")
        ax.set_ylabel("Dependent node")
        ax.set_xticks(range(0, n_nodes, max(1, n_nodes // 6)))
        ax.set_yticks(range(0, n_nodes, max(1, n_nodes // 6)))
        if true_edges:
            xs = [v for u, v in true_edges]
            ys = [u for u, v in true_edges]
            ax.scatter(xs, ys, s=8, c="white", edgecolors="black", linewidths=0.35, label="True edge")
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    assert image is not None
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Edge weight")
    fig.suptitle("Candidate Edge Weights")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cost_curve(
    edge_weight_thresholds: np.ndarray,
    prior_costs: np.ndarray,
    learned_costs: np.ndarray,
    true_cost: float,
    learned_recall: np.ndarray,
    selected_edge_weight_threshold: float,
    output_path: Path,
    recall_guardrail: float = 0.90,
) -> None:
    """Write the two-panel revalidation-cost and recall-guardrail figure."""
    plt = load_pyplot()
    fig, (ax, recall_ax) = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)

    shade_recall_region(ax, edge_weight_thresholds, learned_recall, recall_guardrail)
    ax.plot(edge_weight_thresholds, prior_costs, label="Initial prior graph")
    ax.plot(edge_weight_thresholds, learned_costs, label="Learned graph")
    ax.axhline(true_cost, color="black", linestyle="--", linewidth=1.25, label="True hidden graph")
    ax.axvline(
        selected_edge_weight_threshold,
        color="#9467bd",
        linestyle=":",
        linewidth=1.6,
        label="Selected edge-weight threshold",
    )
    ax.set_xlabel("Edge-weight threshold")
    ax.set_ylabel("Mean routines revalidated per drift")
    ax.set_title("Estimated Revalidation Cost")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.annotate(
        "High edge-weight threshold\ncan miss revalidations.",
        xy=(0.74, 0.12),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#999999", "alpha": 0.88},
    )

    shade_recall_region(recall_ax, edge_weight_thresholds, learned_recall, recall_guardrail)
    recall_ax.plot(edge_weight_thresholds, learned_recall, color="#2ca02c", label="Learned edge recall")
    recall_ax.axhline(
        recall_guardrail,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label=f"Recall guardrail = {recall_guardrail:.2f}",
    )
    recall_ax.axvline(selected_edge_weight_threshold, color="#9467bd", linestyle=":", linewidth=1.6)
    recall_ax.set_xlabel("Edge-weight threshold")
    recall_ax.set_ylabel("Dependency edge recall")
    recall_ax.set_title("Recall Guardrail")
    recall_ax.set_ylim(-0.02, 1.02)
    recall_ax.grid(True, alpha=0.25)
    recall_ax.legend(loc="best")

    fig.suptitle("Estimated Revalidation Cost vs Edge-Weight Threshold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def sorted_weight_panel(
    ax: plt.Axes, labels: np.ndarray, initial_weights: np.ndarray, learned_weights: np.ndarray
) -> None:
    """Draw sorted true-edge and false-edge weights on one summary panel."""
    series = [
        ("True edges, prior", initial_weights[labels == 1], "#1f77b4", "--"),
        ("True edges, learned", learned_weights[labels == 1], "#1f77b4", "-"),
        ("False edges, prior", initial_weights[labels == 0], "#d62728", "--"),
        ("False edges, learned", learned_weights[labels == 0], "#d62728", "-"),
    ]

    for name, values, color, linestyle in series:
        if len(values) == 0:
            continue
        sorted_values = np.sort(values)[::-1]
        x_values = np.linspace(0.0, 1.0, len(sorted_values))
        ax.plot(x_values, sorted_values, label=name, color=color, linestyle=linestyle, linewidth=1.8)

    ax.set_xlabel("Edges sorted within class")
    ax.set_ylabel("Candidate dependency weight")
    ax.set_title("(2) Weight Separation")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_proposal_summary_figure(
    labels: np.ndarray,
    initial_weights: np.ndarray,
    learned_weights: np.ndarray,
    edge_weight_thresholds: np.ndarray,
    prior_costs: np.ndarray,
    learned_costs: np.ndarray,
    true_cost: float,
    prior_recall: np.ndarray,
    learned_recall: np.ndarray,
    selected_edge_weight_threshold: float,
    output_path: Path,
    recall_guardrail: float = 0.90,
) -> None:
    """Write the 2x2 proposal summary figure.

    Panel numbering matches the notebook text: edge precision/recall, weight
    separation, revalidation cost, and recall guardrail.
    """
    plt = load_pyplot()
    initial_pr_recall, initial_precision, initial_auc = pr_curve(labels, initial_weights)
    learned_pr_recall, learned_precision, learned_auc = pr_curve(labels, learned_weights)

    fig, axes_grid = plt.subplots(2, 2, figsize=(12.0, 8.4))
    axes = axes_grid.ravel()

    ax = axes[0]
    ax.plot(initial_pr_recall, initial_precision, label=f"Initial prior ({initial_auc:.3f})")
    ax.plot(learned_pr_recall, learned_precision, label=f"Learned ({learned_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(1) Edge Precision-Recall")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(title="AUC-PR", loc="best", fontsize=8, title_fontsize=8)

    sorted_weight_panel(axes[1], labels, initial_weights, learned_weights)

    ax = axes[2]
    shade_recall_region(ax, edge_weight_thresholds, learned_recall, recall_guardrail)
    ax.plot(edge_weight_thresholds, prior_costs, label="Initial prior graph")
    ax.plot(edge_weight_thresholds, learned_costs, label="Learned graph")
    ax.axhline(true_cost, color="black", linestyle="--", linewidth=1.25, label="True hidden graph")
    ax.axvline(
        selected_edge_weight_threshold,
        color="#9467bd",
        linestyle=":",
        linewidth=1.5,
        label="Selected edge-weight threshold",
    )
    ax.set_xlabel("Edge-weight threshold")
    ax.set_ylabel("Mean routines revalidated per drift")
    ax.set_title("(3) Estimated Revalidation Cost")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.98,
        0.04,
        "High edge-weight threshold\ncan miss revalidations.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#aaaaaa", "alpha": 0.9},
    )

    ax = axes[3]
    shade_recall_region(ax, edge_weight_thresholds, learned_recall, recall_guardrail)
    ax.plot(edge_weight_thresholds, prior_recall, label="Initial prior recall")
    ax.plot(edge_weight_thresholds, learned_recall, label="Learned recall")
    ax.axhline(
        recall_guardrail,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label=f"Recall guardrail = {recall_guardrail:.2f}",
    )
    ax.axvline(
        selected_edge_weight_threshold,
        color="#9467bd",
        linestyle=":",
        linewidth=1.5,
        label="Selected edge-weight threshold",
    )
    ax.set_xlabel("Edge-weight threshold")
    ax.set_ylabel("Dependency edge recall")
    ax.set_title("(4) Recall Guardrail")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle("Toy Learning of Calibration Dependency Edges", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def config_for_json(config: SimulationConfig) -> dict[str, object]:
    """Convert config to JSON-serializable values."""
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    return data


def run_simulation(config: SimulationConfig) -> dict[str, object]:
    """Run the full synthetic experiment and write plots/metrics.

    This is the main programmatic entry point. It returns the same metrics that
    are written to metrics.json, making it convenient for notebooks or tests to
    inspect results without reparsing the file.
    """
    rng = np.random.default_rng(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    true_graph, true_edges, _, edges, distances = generate_hidden_true_dag(config, rng)
    initial_weights, known_edges, local_mask = create_prior_weights(
        config, rng, edges, true_edges, distances
    )
    learned_weights = initial_weights.copy()

    # Repeated synthetic observations act like pre-calibration evidence about
    # whether the candidate weighted graph matches the hidden dependency graph.
    for _ in range(config.n_events):
        root, affected, cause_parent = simulate_hidden_event(true_graph, config, rng)
        observed_mask, noisy_state = simulate_check_data_observations(config, affected, root, rng)
        learned_weights = update_weights_from_observations(
            learned_weights,
            edges,
            observed_mask,
            noisy_state,
            cause_parent,
            config,
            rng,
        )

    labels = np.array([1 if edge in true_edges else 0 for edge in edges], dtype=int)
    initial_metrics = edge_recall_metrics(labels, initial_weights)
    learned_metrics = edge_recall_metrics(labels, learned_weights)

    cost_rng = np.random.default_rng(config.seed + 10_001)
    edge_weight_thresholds, prior_costs, learned_costs, true_cost, roots = cost_curve(
        config, true_graph, edges, initial_weights, learned_weights, cost_rng
    )
    prior_tradeoff = edge_weight_threshold_tradeoff_curve(
        config, edges, labels, initial_weights, roots, edge_weight_thresholds
    )
    learned_tradeoff = edge_weight_threshold_tradeoff_curve(
        config, edges, labels, learned_weights, roots, edge_weight_thresholds
    )
    recall_guardrail = 0.90
    selected_operating_point = select_edge_weight_threshold(
        learned_tradeoff, recall_guardrail=recall_guardrail
    )

    prior_graph_05 = graph_from_scores(config.n_nodes, edges, initial_weights, 0.5)
    learned_graph_05 = graph_from_scores(config.n_nodes, edges, learned_weights, 0.5)
    learned_graph_best = graph_from_scores(
        config.n_nodes, edges, learned_weights, learned_metrics["best_edge_weight_threshold"]
    )
    revalidation_cost_prior_at_0_5 = mean_revalidation_cost(prior_graph_05, roots)
    revalidation_cost_learned_at_0_5 = mean_revalidation_cost(learned_graph_05, roots)

    metrics: dict[str, object] = {
        "initial_auc_pr": initial_metrics["auc_pr"],
        "learned_auc_pr": learned_metrics["auc_pr"],
        "initial_best_f1": initial_metrics["best_f1"],
        "learned_best_f1": learned_metrics["best_f1"],
        "revalidation_cost_prior_at_edge_weight_threshold_0_5": revalidation_cost_prior_at_0_5,
        "revalidation_cost_learned_at_edge_weight_threshold_0_5": revalidation_cost_learned_at_0_5,
        "revalidation_cost_true": true_cost,
        "selected_edge_weight_threshold": selected_operating_point[
            "selected_edge_weight_threshold"
        ],
        "selected_edge_weight_threshold_metrics": selected_operating_point,
        "edge_weight_threshold_curves": {
            "initial_prior": prior_tradeoff,
            "learned": learned_tradeoff,
        },
        "config": config_for_json(config),
        "graph_summary": {
            "edge_convention": "u -> v means calibration routine u depends on v; valid edges require u > v.",
            "n_candidate_edges": len(edges),
            "n_true_edges": len(true_edges),
            "n_known_prior_edges": len(known_edges),
            "n_spatial_local_candidate_edges": int(np.sum(local_mask)),
            "true_edge_density": len(true_edges) / len(edges) if edges else 0.0,
        },
        "edge_recall": {
            "initial_prior": initial_metrics,
            "learned": learned_metrics,
            "auc_pr_gain": learned_metrics["auc_pr"] - initial_metrics["auc_pr"],
            "best_f1_gain": learned_metrics["best_f1"] - initial_metrics["best_f1"],
        },
        "revalidation_cost": {
            "cost_model": (
                "For each drift root, revalidation cost is the drifted node plus all "
                "transitive dependents under the edge-weight-thresholded graph. This "
                "metric is a proxy for validation work, not a hardware timing or "
                "pulse-level recalibration measurement."
            ),
            "n_eval_drifts": int(len(roots)),
            "edge_weight_threshold_0_5": {
                "true_graph": true_cost,
                "initial_prior_graph": revalidation_cost_prior_at_0_5,
                "learned_graph": revalidation_cost_learned_at_0_5,
            },
            "learned_best_f1_edge_weight_threshold": {
                "edge_weight_threshold": learned_metrics["best_edge_weight_threshold"],
                "learned_graph": mean_revalidation_cost(learned_graph_best, roots),
                "true_graph": true_cost,
            },
            "selected_edge_weight_threshold": selected_operating_point,
        },
        "sklearn_available": SKLEARN_AVAILABLE,
        "outputs": {
            "edge_precision_recall_curve": str(config.output_dir / "edge_precision_recall_curve.png"),
            "edge_weight_heatmap_before_after": str(
                config.output_dir / "edge_weight_heatmap_before_after.png"
            ),
            "cost_recall_tradeoff": str(config.output_dir / "cost_recall_tradeoff.png"),
            "proposal_summary_figure": str(config.output_dir / "proposal_summary_figure.png"),
            "metrics_json": str(config.output_dir / "metrics.json"),
        },
    }

    plot_pr_curves(labels, initial_weights, learned_weights, config.output_dir / "edge_precision_recall_curve.png")
    plot_weight_heatmaps(
        config.n_nodes,
        edges,
        true_edges,
        initial_weights,
        learned_weights,
        config.output_dir / "edge_weight_heatmap_before_after.png",
    )
    plot_cost_curve(
        edge_weight_thresholds,
        prior_costs,
        learned_costs,
        true_cost,
        np.array(learned_tradeoff["edge_recall"], dtype=float),
        float(selected_operating_point["selected_edge_weight_threshold"]),
        config.output_dir / "cost_recall_tradeoff.png",
        recall_guardrail=recall_guardrail,
    )
    plot_proposal_summary_figure(
        labels,
        initial_weights,
        learned_weights,
        edge_weight_thresholds,
        prior_costs,
        learned_costs,
        true_cost,
        np.array(prior_tradeoff["edge_recall"], dtype=float),
        np.array(learned_tradeoff["edge_recall"], dtype=float),
        float(selected_operating_point["selected_edge_weight_threshold"]),
        config.output_dir / "proposal_summary_figure.png",
        recall_guardrail=recall_guardrail,
    )

    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    return metrics


def parse_args() -> SimulationConfig:
    """Parse command-line arguments into a SimulationConfig."""
    parser = argparse.ArgumentParser(
        description="Toy simulation for learning calibration dependency DAG weights."
    )
    parser.add_argument("--n-nodes", type=int, default=30, help="Number of calibration nodes.")
    parser.add_argument("--n-events", type=int, default=600, help="Number of simulated drift events.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--p-true", type=float, default=0.08, help="Average probability for hidden true DAG edges."
    )
    parser.add_argument(
        "--obs-noise", type=float, default=0.05, help="False check_data observation probability."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Directory for plots and metrics."
    )
    parser.add_argument(
        "--no-spatial-locality",
        action="store_true",
        help="Disable spatial locality in hidden edge generation.",
    )
    parser.add_argument(
        "--propagation-prob",
        type=float,
        default=0.80,
        help="Probability that badness propagates across each true dependency edge.",
    )
    args = parser.parse_args()

    if args.n_nodes < 2:
        raise SystemExit("--n-nodes must be at least 2")
    if args.n_events < 1:
        raise SystemExit("--n-events must be at least 1")
    if not 0.0 <= args.p_true <= 1.0:
        raise SystemExit("--p-true must be between 0 and 1")
    if not 0.0 <= args.obs_noise <= 0.5:
        raise SystemExit("--obs-noise must be between 0 and 0.5")
    if not 0.0 <= args.propagation_prob <= 1.0:
        raise SystemExit("--propagation-prob must be between 0 and 1")

    return SimulationConfig(
        n_nodes=args.n_nodes,
        n_events=args.n_events,
        seed=args.seed,
        p_true=args.p_true,
        obs_noise=args.obs_noise,
        output_dir=args.output_dir,
        spatial_locality=not args.no_spatial_locality,
        propagation_prob=args.propagation_prob,
    )


def main() -> None:
    """CLI entry point."""
    config = parse_args()
    metrics = run_simulation(config)

    initial = metrics["edge_recall"]["initial_prior"]  # type: ignore[index]
    learned = metrics["edge_recall"]["learned"]  # type: ignore[index]
    revalidation_cost = metrics["revalidation_cost"]["edge_weight_threshold_0_5"]  # type: ignore[index]
    selected = metrics["selected_edge_weight_threshold_metrics"]  # type: ignore[index]

    print("Toy calibration dependency learning complete")
    print(f"Outputs: {config.output_dir}")
    print("Final report:")
    print(f"  selected edge-weight threshold: {selected['selected_edge_weight_threshold']:.2f}")
    print(f"  precision: {selected['precision']:.3f}")
    print(f"  recall: {selected['recall']:.3f}")
    print(f"  F1: {selected['f1']:.3f}")
    print(f"  revalidation cost: {selected['revalidation_cost']:.2f}")
    print(f"  initial AUC-PR: {initial['auc_pr']:.3f}")
    print(f"  learned AUC-PR: {learned['auc_pr']:.3f}")
    print(
        "AUC-PR: "
        f"initial={initial['auc_pr']:.3f}, learned={learned['auc_pr']:.3f}, "
        f"gain={metrics['edge_recall']['auc_pr_gain']:.3f}"  # type: ignore[index]
    )
    print(
        "F1@0.5: "
        f"initial={initial['f1_at_0_5']:.3f}, learned={learned['f1_at_0_5']:.3f}; "
        f"best learned F1={learned['best_f1']:.3f} "
        f"at edge-weight threshold={learned['best_edge_weight_threshold']:.2f}"
    )
    print(
        "Mean revalidation cost @0.5: "
        f"true={revalidation_cost['true_graph']:.2f}, "
        f"prior={revalidation_cost['initial_prior_graph']:.2f}, "
        f"learned={revalidation_cost['learned_graph']:.2f}"
    )
    print(
        "Selected edge-weight threshold: "
        f"{selected['selected_edge_weight_threshold']:.2f} "
        f"(precision={selected['precision']:.3f}, recall={selected['recall']:.3f}, "
        f"recall guardrail={selected['recall_guardrail']:.2f}, "
        f"F1={selected['f1']:.3f}, revalidation cost={selected['revalidation_cost']:.2f})"
    )


if __name__ == "__main__":
    main()
