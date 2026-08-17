"""
Statistical significance testing for model comparisons.

Implements McNemar's test for paired binary classifier predictions, as
referenced in the manuscript for comparing HeartGuard Pro against
baselines (CNN-only, standard BiLSTM, transformer-only, InceptionTime).

IMPORTANT: p-values here are computed from whatever prediction arrays
you pass in. To honestly reproduce the manuscript's comparison table,
you must actually train each baseline (see src/models/baselines.py)
on the same data split and pass in their real predictions -- not
numbers copied from the current draft.
"""
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray,
                  preds_b: np.ndarray, exact: bool = True) -> dict:
    """Paired McNemar test comparing two classifiers' correctness on
    the same test set.

    Parameters
    ----------
    y_true : ground truth labels
    preds_a, preds_b : predictions from model A (e.g. HeartGuard Pro)
        and model B (a baseline), same length/order as y_true.

    Returns
    -------
    dict with the 2x2 contingency table and the McNemar p-value.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, a_only],
             [b_only, both_wrong]]

    result = mcnemar(table, exact=exact)
    return {
        "contingency_table": table,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
    }


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot: int = 2000,
                  ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap confidence interval for any sklearn-style metric
    function, e.g. bootstrap_ci(f1_score, y_true, y_pred).

    Use this instead of writing down a plausible-looking "± x.xxx"
    by hand -- it computes the interval from resampling your actual
    predictions.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(scores, [alpha, 1 - alpha])
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


if __name__ == "__main__":
    # Sanity check with synthetic data -- NOT a manuscript result.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    pred_a = np.where(rng.random(500) < 0.9, y, 1 - y)   # 90% acc model
    pred_b = np.where(rng.random(500) < 0.85, y, 1 - y)  # 85% acc model
    print("Synthetic sanity check (not a real result):")
    print(mcnemar_test(y, pred_a, pred_b))
