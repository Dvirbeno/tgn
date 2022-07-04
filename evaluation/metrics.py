import numpy as np
from scipy import stats
import pandas as pd
from sklearn import metrics


def calc_metrics(preds: np.ndarray, targets: np.ndarray):
    assert preds.shape[0] == targets.shape[0]
    N = preds.shape[0]

    ranked_preds = (pd.Series(preds).rank(pct=True).values - (1 / N)) * N / (N - 1)
    ranked_targets = (pd.Series(targets).rank(pct=True).values - (1 / N)) * N / (N - 1)

    if len(np.unique(ranked_preds)) > 1:
        spearmanr, _ = stats.spearmanr(ranked_preds, ranked_targets)
        ktau, _ = stats.kendalltau(ranked_preds, ranked_targets)
    else:
        spearmanr = np.nan
        ktau = np.nan
    mae = metrics.mean_absolute_error(ranked_targets, ranked_preds)

    desc_order = np.argsort(-ranked_preds)
    sorted_preds = ranked_preds[desc_order]
    sorted_targets = ranked_targets[desc_order]
    rank_sorted_targets = pd.Series(sorted_targets).rank(ascending=False).values - 1
    relevance_scores = 1. / (1. + np.abs(sorted_targets - sorted_preds))

    mrr = np.mean(relevance_scores)
    precision_times_relevance = []
    for i, relevance in enumerate(relevance_scores):
        precision = np.mean(rank_sorted_targets[:i + 1] <= i)
        precision_times_relevance.append(precision * relevance)
    map = np.mean(precision_times_relevance)

    dcg = np.sum((1 / (np.log2(1 + np.arange(1, N + 1)))) * relevance_scores)
    idcg = np.sum((1 / (np.log2(1 + np.arange(1, N + 1)))))
    ndcg = dcg / idcg

    return {'spearmanr': spearmanr,
            'ktau': ktau,
            'mae': mae,
            'mrr': mrr,
            'map': map,
            'dcg': dcg,
            'ndcg': ndcg}
