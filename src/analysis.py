# -*- coding: utf-8 -*-

"""Script containing code for analysis."""

import itertools as itt
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.constants import PREDICTIONS_PATH, DRUG_PREFIXES
from .models import get_model_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_normalized(df) -> bool:
    if np.min(df['score']) < 0:
        return False
    return np.max(df['score']) <= 1


def get_models_intersection(df, K_values):
    fields = ['source', 'target', 'score']
    model_topologies = df.model.unique()
    results = df[['kg', 'model'] + fields].copy()

    data = []
    for kg in results['kg'].unique():
        results_kg = results[results['kg'] == kg]
        for K in K_values:
            for model1, model2 in itt.product(model_topologies, model_topologies):
                if model1 == model2:
                    data.append([kg, K, model1, model2, 0, K])
                    continue

                df_model1 = results_kg[results_kg['model'] == model1][fields].sort_values('score',
                                                                                          ascending=False).head(K)
                df_model2 = results_kg[results_kg['model'] == model2][fields].sort_values('score',
                                                                                          ascending=False).head(K)

                df = pd.concat([df_model1, df_model2], ignore_index=True)
                #             preds = df[~df.set_index(['source', 'target']).index.duplicated()].reset_index()
                preds = df.groupby(['source', 'target'])
                data.append([kg, K, model1, model2, len(df) - len(preds), len(preds)])

            df = results_kg[fields].head(K * len(model_topologies)).sort_values('score', ascending=False)
            #         preds = df[~df.set_index(['source', 'target']).index.duplicated()].reset_index()
            preds = df.groupby(['source', 'target'])
            data.append([kg, K, 'all', 'all', len(df) - len(preds), len(preds)])

    headers = ['kg', 'K', 'model1', 'model2', 'intersection', 'union']
    models_intersection = pd.DataFrame(data, columns=headers)
    return models_intersection


def get_trials_intersection(df, max_trials=None):
    fields = ['source', 'target', 'score']
    model_topologies = df.model.unique()
    results = df[['kg', 'model', 'trial_k'] + fields]

    if max_trials is None:
        trials = results['trial_k'].unique()
    else:
        trials = range(max_trials)

    data = []
    for kg in results['kg'].unique():
        results_kg = results[results['kg'] == kg]
        for K in [1, 10, 100, 1000]:
            for model in model_topologies:
                results_model = results_kg[results_kg['model'] == model]

                for trial_k1, trial_k2 in itt.product([0, 1, 2], [0, 1, 2]):
                    if trial_k1 == trial_k2:
                        data.append([kg, model, K, trial_k1, trial_k2, 0, K])
                        continue
                    df = pd.concat([
                        results_model[results_model['trial_k'] == trial_k].head(K)[fields]
                        for trial_k in [trial_k1, trial_k2]
                    ], ignore_index=True).reset_index(drop=True).copy()
                    preds = df.groupby(['source', 'target'])
                    data.append([kg, model, K, trial_k1, trial_k2, len(df) - len(preds), len(df)])

                df = results_model[fields].sort_values('score', ascending=False).head(K * 3).reset_index(
                    drop=True).copy()
                preds = df.groupby(['source', 'target'])
                data.append([kg, model, K, 'all', 'all', len(df) - len(preds), len(df)])

    headers = ['kg', 'model', 'K', 'trial1', 'trial2', 'intersection', 'union']
    trials_intersection = pd.DataFrame(data, columns=headers)
    return trials_intersection


def cherry_pick(predictions, topk=10):
    models = [m for m in predictions.model.unique() if not m.startswith('multi')]
    multi_models = [m for m in predictions.model.unique() if m.startswith('multi')]

    kgs = predictions.kg.unique()

    top_predictions = {kg: {} for kg in kgs}
    columns = ['kg', 'multi-model', 'source', 'target', 'position']
    columns.extend([m for m in models])
    data = []

    for kg in kgs:
        df = predictions[predictions['kg'] == kg]
        for multi_model in multi_models:
            df_multi = df[df['model'] == multi_model]
            df_multi = df_multi.sort_values('score').reset_index().head(topk)
            top_predictions[kg][multi_model] = df_multi.copy()
            for new_pos, (source, target, y) in enumerate(
                top_predictions[kg][multi_model]['source, target', 'y'].values):
                if not y:
                    continue

                data.append([kg, multi_model, source, target, new_pos])
                for model in models:
                    df_ = df[df['model'] == model]
                    df_ = df.sort_values('score').reset_index()

                    old_pos = df_[(df_['source'] == source) & (df_['target'] == target)].index
                    data[-1].extend(old_pos)

    picks = pd.DataFrame(data, columns=columns)
    return picks


def add_ground_truth(predictions):
    fields = ['source', 'target', 'score']

    model_data = get_model_data()

    predictions_gt = pd.DataFrame([])
    for kg, drug_prefix in DRUG_PREFIXES.items():
        golden = model_data[
            (model_data['kg'] == kg) & \
            (model_data['dataset'] == 'test')][['source', 'target']]
        golden['y'] = 1
        model_topologies = predictions[predictions['kg'] == kg]['model'].unique()
        for model_topo in model_topologies:
            trials = predictions['trial_k'].unique()
            for trial_k in trials:
                preds = predictions[
                    (predictions['kg'] == kg) & \
                    (predictions['model'] == model_topo) & \
                    (predictions['trial_k'] == trial_k)
                    ][fields]

                preds = preds[preds['source'].str.startswith(drug_prefix)]
                preds = pd.merge(preds, golden, how='left', on=['source', 'target'])
                preds['y'] = preds['y'].apply(lambda x: 0 if np.isnan(x) else 1)
                preds['kg'] = kg
                preds['model'] = model_topo
                preds['trial_k'] = trial_k
                predictions_gt = pd.concat([predictions_gt, preds], ignore_index=True)

    return predictions_gt


def get_model_precision(df, kg, model, K_values, golden):
    data = []
    total_tps = len(golden)

    df_model = df[
        (df['kg'] == kg) & \
        (df['model'] == model)
        ].sort_values('score', ascending=False)
    for K in K_values:
        df_k = df_model.head(K)
        tps_in_k = len(df_k[df_k['y'] == True])

        precision = (tps_in_k / K) * 100
        data.append([
            kg, model, K, precision, tps_in_k, total_tps
        ])

    headers = ['kg', 'model', 'K', 'precision', 'tps_in_k', 'total_tps']
    precision_data = pd.DataFrame(data, columns=headers)
    precision_data = precision_data.sort_values('K', ascending=False)
    return precision_data


def get_precision(df, K_values, dataset='test', max_trial_k=1, num_procs=4):
    assert max_trial_k == 1

    model_data = get_model_data()
    model_data = model_data[model_data['dataset'] == dataset].copy().reset_index()

    models = df.model.unique()
    kgs = df.kg.unique()

    golden = {}
    for kg in kgs:
        golden[kg] = model_data[model_data['kg'] == kg][['source', 'target']].drop_duplicates(keep="first")

    params = [
        [df, kg, model, K_values, golden[kg]]
        for kg, model in itt.product(kgs, models)
        if not os.path.exists(os.path.join(PREDICTIONS_PATH, f'precision_{kg}_{model}-{dataset}.csv'))
    ]

    to_read = [
        [kg, model, os.path.join(PREDICTIONS_PATH, f'precision_{kg}_{model}-{dataset}.csv')]
        for kg, model in itt.product(kgs, models)
        if os.path.exists(os.path.join(PREDICTIONS_PATH, f'precision_{kg}_{model}-{dataset}.csv'))
    ]

    logger.info(f"Computing {dataset} precision for {models}.")

    precision_data = pd.DataFrame([])
    for kg, model, fn in to_read:
        df = pd.read_csv(fn, sep='\t')
        df['kg'] = kg
        df['model'] = model
        precision_data = pd.concat([precision_data, df], ignore_index=True)
        print(df.columns)

    if len(params) > 0:
        with Pool(processes=min(num_procs, len(params))) as pool:
            results = pool.starmap(get_model_precision, params)

            for model_precision in results:
                model = model_precision.model.unique()[0]
                kg = model_precision.kg.unique()[0]
                logger.info(f"Computed {dataset} precision for {model} in {kg}.")

                precision_csv = os.path.join(PREDICTIONS_PATH, f'precision_{kg}_{model}-{dataset}.csv')
                model_precision.to_csv(precision_csv, sep='\t', index=False)
                precision_data = pd.concat([precision_data, model_precision], ignore_index=True)

    return precision_data
