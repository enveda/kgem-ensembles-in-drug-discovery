# -*- coding: utf-8 -*-

"""This module implements methods used to create/combine the ensemble of models."""

import math
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import analysis


def normalize_quantiles(df):
    assert len(df['kg'].unique() == 1)
    df_ = df[['source', 'target', 'model', 'score']].copy().reset_index(drop=True)
    df_ = df_.set_index(['source', 'target'])
    pivot = pd.pivot_table(
        df_,
        values='score',
        index=['source', 'target'],
        columns=['model'],
    )

    rank_mean = pivot.stack().groupby(df.rank(method='first').stack().astype(float)).mean()
    df_.rank(method='min').stack().astype(float).map(rank_mean).unstack()


def normalize_scores(df):
    if len(df) == 0:
        return df
    min_score = min(df["score"])
    max_score = max(df["score"])
    assert len(df) > 1
    df["score"] = df["score"].apply(lambda x: (x - min_score) / (max_score - min_score))

    min_norm_score = min(df["score"])
    max_norm_score = max(df["score"])
    assert min_norm_score == 0 and max_norm_score == 1
    return df


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def normalize_sigmoid(df):
    df_ = df.copy()
    df_['score'] = df_['score'].apply(lambda x: sigmoid(x))
    return df_


def normalize(
    scores_dict: dict,
    min_value: float
):
    """Normalize the scores between 0 and 1"""
    normalize_score_paths = defaultdict(list)

    for key in scores_dict:
        for idx, path in enumerate(scores_dict[key]):
            norm_scores = {}

            for node in path:
                val = 1 - (scores_dict[key][idx][node] / min_value)
                norm_scores[node] = round(val.item(), 3)

            normalize_score_paths[key].append(norm_scores)

    return normalize_score_paths


def shift_scores(df):
    df_ = df.copy()
    min_score = np.min(df_['score'].values)
    df_['score'] = df_['score'].apply(lambda x: x - min_score)
    assert np.min(df_['score'].values) == 0
    return df_


def center_scores(df):
    df_ = df.copy()
    min_score = np.min(df_['score'].values)
    max_score = np.max(df_['score'].values)
    center = (max_score - min_score) / 2
    if center < 0:
        center = -center
    df_['score'] = df_['score'] - center
    return df_


def avg_center_scores(df):
    df_ = df.copy()
    avg = np.average(df_['score'].values)
    if avg < 0:
        avg = -avg
    df['score'] = df['score'] - avg
    return df_


def apply_normalization(df, norm_fn="position"):
    # import pdb; pdb.set_trace()
    if len(df['model'].unique()) > 1:
        raise ValueError("trying to normalize DataFrame with multiple models.")
    if len(df['kg'].unique()) > 1:
        raise ValueError("trying to normalize DataFrame with multiple kgs.")

    if norm_fn == "scores":
        return normalize_scores(df)
    elif "sigmoid" in norm_fn:
        if "shifted" in norm_fn:
            df = shift_scores(df)
        elif "0centered" in norm_fn:
            df = center_scores(df)
        elif "avgcentered" in norm_fn:
            df = avg_center_scores(df)
        elif "norm" in norm_fn:
            df = normalize_scores(df)
        return normalize_sigmoid(df)
    elif norm_fn == "position":
        num_pairs = len(df)
        df['score'] = df.reset_index().index
        df['score'] = df['score'].apply(lambda x: float((num_pairs - x) / num_pairs))
        return df

    raise ValueError


def combine_scores(
    predictions, norm_fn="position",
    norm_topk=None, weights=None,
    use_k_best_trials=1
):
    fields = ['source', 'target', 'score', 'y']
    joint_preds = pd.DataFrame([], columns=['kg', 'trial_k'] + fields)

    for kg in predictions['kg'].unique():
        kg_preds = []
        for model in predictions.model.unique():

            df = predictions[
                (predictions['kg'] == kg) & \
                (predictions['model'] == model) & \
                (predictions['trial_k'] < use_k_best_trials)
                ].copy().reset_index(drop=True)
            df = df.sort_values('score', ascending=False)

            topk_name = 'all'
            if norm_topk is not None:
                if norm_topk >= 1:
                    topk_name = f'top{norm_topk}'
                    df = df.head(norm_topk).copy().reset_index()
                else:
                    percentile = norm_topk * 100
                    topk_name = f'perc{percentile}'
                    min_score = np.percentile(df['score'].values, percentile)
                    df = df[df['score'] >= min_score]
            df = apply_normalization(df, norm_fn=norm_fn)

            df = df[fields].copy()
            if weights is not None:
                df['score'] = df['score'] * weights[kg][model]
            kg_preds.append(df)

        df = pd.concat(kg_preds).groupby(['source', 'target'], as_index=False)
        df = df[['score', 'y']].sum()
        df = df.sort_values('score', ascending=False)
        df['y'] = df['y'].apply(lambda x: int(x) > 0)
        df['kg'] = kg
        df['trial_k'] = use_k_best_trials - 1
        ensemble_name = f'multi-{norm_fn}-{topk_name}{"-weighted" if weights is not None else ""}'
        df['model'] = ensemble_name
        joint_preds = pd.concat([joint_preds, df], ignore_index=True)

    return joint_preds


def product_of_experts(df):
    candidates = df.copy()
    candidates['score'] = candidates['score'].apply(lambda x: sigmoid(x))
    candidates = candidates.groupby(['source', 'target']).prod().reset_index()

    sum_of_candidates = np.sum(candidates['score'].values)
    candidates['score'] = candidates['score'].apply(lambda x: x / sum_of_candidates)
    return candidates


def compute_weights(precision_data, K, dataset='val'):
    models = precision_data.model.unique()

    df_pr = precision_data
    if 'dataset' in precision_data.columns:
        df_pr = df_pr[df_pr['dataset'] == dataset]
    df_pr = df_pr[df_pr['K'] == K].copy().reset_index()

    weights = {}
    for kg in df_pr.kg.unique():
        weights[kg] = {}
        total_weight = df_pr[(df_pr['kg'] == kg)]['precision'].sum()
        if total_weight == 0:
            print(f'total weight == 0 for {kg} and K={K}')
        for model in models:
            try:
                w = df_pr[(df_pr['kg'] == kg) & (df_pr['model'] == model)]['precision'].values[0]
            except Exception as e:
                print(f'{model} with {kg}')
                raise e
            weights[kg][model] = 1 if total_weight == 0 else w / total_weight

    return weights


def combine_by_position_and_scores(predictions, weights=None):
    combined_by_position = combine_scores(predictions, norm_fn='position', weights=weights)
    combined_by_scores = combine_scores(predictions, norm_fn='scores', weights=weights)
    combined_predictions = pd.concat([combined_by_position, combined_by_scores], ignore_index=True)
    return combined_predictions


def exploration_of_score_combination(
    predictions,
    precision_data=None,
    weights=None,
    K_values=[1, 5, 10, 25, 50, 100, 500],
):
    combined_predictions = combine_by_position_and_scores(predictions)

    if precision_data is None:
        precision_data = analysis.get_precision(predictions, K_values)

    combined_precision_data = analysis.get_precision(combined_predictions, K_values)
    combined_precision_data['model'] = combined_precision_data['model'].apply(lambda x: f'{x}-raw')
    for K in K_values:
        weights = compute_weights(precision_data, K)
        combined_weighted_predictions = combine_scores(
            predictions, norm_fn="scores", weights=weights,
        )
        precision_at_K = analysis.get_precision(
            combined_weighted_predictions, K_values=[K],
        )
        precision_at_K['model'] = precision_at_K['model'].apply(lambda x: f'{x}-adjusted@K')

        precision_all_at_K = analysis.get_precision(
            combined_weighted_predictions, K_values=K_values,
        )
        precision_all_at_K['model'] = precision_all_at_K['model'].apply(lambda x: f'{x}-all@{K}')
        combined_precision_data = pd.concat(
            [combined_precision_data, precision_at_K, precision_all_at_K],
            ignore_index=True,
        )

    precision_data = pd.concat(
        [precision_data, combined_precision_data],
        ignore_index=True
    )
    return precision_data


def combine_scores_mp(
    predictions: pd.DataFrame,
    experiment_id: str,
    norm_fn: str = None,
    norm_topk: str = None,
    weights: dict = None
) -> pd.DataFrame:
    combined_df = combine_scores(
        predictions=predictions,
        norm_fn=norm_fn,
        norm_topk=norm_topk,
        weights=weights,
    )
    combined_df['experiment'] = experiment_id
    return combined_df


def exploration_of_score_normalization(
    predictions: pd.DataFrame,
    weights: dict = None,
    num_procs: int = 4,
) -> pd.DataFrame:
    experiments = {
        'norm_all': ["scores", None, weights],
        'norm_position': ["position", None, weights],
        'norm_top500': ["scores", 500, weights],
        'norm_top1k': ["scores", 1000, weights],
        'norm_top5k': ["scores", 5000, weights],
        'norm_p99.95': ["scores", 0.95, weights],
        'norm_p99': ["scores", 0.99, weights],
        'norm_p99.9': ["scores", 0.999, weights],
        'sigmoid': ["sigmoid", None, weights],
        'sigmoid-norm': ["sigmoid-norm", None, weights],
        'sigmoid-norm': ["sigmoid-norm-p99", 0.99, weights],
        'sigmoid-shifted': ["sigmoid-shifted", None, weights],
        'sigmoid-0centered': ["sigmoid-0centered", None, weights],
        'sigmoid-avgcentered': ["sigmoid-avgcentered", None, weights],
    }

    experiments_mp = [[predictions, exp] + params for exp, params in experiments.items()]

    combined_predictions = pd.DataFrame([])

    with Pool(processes=num_procs) as pool:
        results = pool.starmap(combine_scores_mp, experiments_mp)
        for result in results:
            combined_predictions = pd.concat([combined_predictions, result], ignore_index=True)

    poe = pd.DataFrame([])
    for kg in predictions['kg'].unique():
        df_kg = predictions[predictions['kg'] == kg]
        poe_kg = product_of_experts(df_kg)
        poe_kg['kg'] = kg
        poe = pd.concat([poe, poe_kg], ignore_index=True)

    poe['model'] = 'PoE'
    combined_predictions = pd.concat([combined_predictions, poe], ignore_index=True)
    return combined_predictions


def evaluate_ensemble(
    ensemble_predictions,
    K_values,
    norm_fn="scores",
    norm_topk=None,
    weights=None,
):
    combined_predictions = combine_scores(
        ensemble_predictions,
        norm_fn=norm_fn,
        norm_topk=norm_topk,
        weights=weights,
    )

    ensemble_precision = analysis.get_precision(
        combined_predictions,
        K_values=K_values,
        num_procs=2,
    )

    return ensemble_precision
