# -*- coding: utf-8 -*-

"""This module contains redefinitions of PyKEEN prediction methods to use _ScoreConsumer classes."""

import os
from multiprocessing import Pool

import pandas as pd
import torch
from pykeen.models.predict import get_tail_prediction_df
from pykeen.triples import TriplesFactory
from tqdm.auto import tqdm

from src.constants import PREDICTIONS_PATH
from src.models import (
    get_model_data,
    get_model_labels,
    get_topk_model_paths,
)

predict_columns = ['tail_id', 'tail_label', 'score', 'head_label']
headers = ['source', 'relation', 'target']


def save_raw_predictions(predictions, output_fn):
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    predictions.to_csv(output_fn, sep='\t', index=False)


def predict_mp(model, drug_labels, disease_labels, triples_factory):
    def get_data_to_predict():
        for drug in tqdm(drug_labels, desc="preds"):
            yield model, drug, disease_labels, triples_factory

    model_preds = pd.DataFrame([])
    with Pool(processes=4) as pool:
        preds = pool.starmap(predict_single, get_data_to_predict())
        model_preds = pd.concat([model_preds, preds], ignore_index=True)

    model_preds = model_preds.sort_values('score', ascending=False)
    return model_preds


def predict_single(model, head_label, disease_labels, triples_factory):
    preds = get_tail_prediction_df(
        model=model,
        head_label=head_label,
        relation_label='treats',
        triples_factory=triples_factory,
        add_novelties=True,
        remove_known=True,
    )

    preds = preds[preds['tail_label'].isin(disease_labels)]
    preds['head_label'] = head_label
    return preds


def predict(model, drug_labels, disease_labels, triples_factory):
    model_preds = pd.DataFrame([], columns=predict_columns)
    for head_label in tqdm(drug_labels, desc="preds"):
        try:
            preds = get_tail_prediction_df(
                model=model,
                head_label=head_label,
                relation_label='treats',
                triples_factory=triples_factory,
                add_novelties=True,
                remove_known=True,
            )
        except Exception as e:
            print(f'Error predicting {head_label} label')
            raise e

        preds = preds[preds['tail_label'].isin(disease_labels)]
        preds['head_label'] = head_label

        model_preds = pd.concat([model_preds, preds], ignore_index=True)

    model_preds = model_preds.sort_values('score', ascending=False)
    return model_preds


def get_all_predictions(
    model_topologies,
    kgs=None,
    top_k_trials=1,
    save=False,
    skip_if_exists=True,
    return_all=False,
    batch_size=1,
    dataset_to_predict='test',
    #     return_only_new=True,
):
    model_data = get_model_data()
    drug_labels, disease_labels = get_model_labels(model_data, dataset_to_predict)
    if kgs is None:
        kg_names = model_data['kg'].unique()
    else:
        kg_names = [kg for kg in model_data['kg'].unique() if kg in kgs]

    topk_models_paths = get_topk_model_paths(top_k=top_k_trials)

    columns = ['kg', 'model', 'trial_k'] + predict_columns
    all_models_predictions = pd.DataFrame([], columns=columns)

    for kg in tqdm(kg_names, desc="KG's"):
        known_datasets = ['train', 'val'] if dataset_to_predict == 'test' else ['train']
        known_triples = model_data[
            (model_data['kg'] == kg) & \
            (model_data['dataset'].isin(known_datasets))
            ][headers].copy()

        triples_factory = None
        for model_topo in tqdm(model_topologies, desc="Models"):
            model_kg = topk_models_paths[kg].get(model_topo, None)
            if model_kg is None:
                print(f'{model_topo} not available for {kg}')
                continue
            max_model_trials = len(model_kg)

            for trial_k in tqdm(range(min(top_k_trials, max_model_trials)), desc="trials"):
                output_fn = f'preds_{kg}_{model_topo}-trial_{trial_k}-{dataset_to_predict}.csv'
                output_fn = os.path.join(PREDICTIONS_PATH, output_fn)

                if skip_if_exists and os.path.exists(output_fn):
                    if return_all:
                        model_preds = pd.read_csv(output_fn, sep='\t')
                        all_models_predictions = pd.concat(
                            [all_models_predictions, model_preds],
                            ignore_index=True)
                    continue

                if triples_factory is None:
                    triples_factory = TriplesFactory.from_labeled_triples(known_triples.to_numpy())

                model_path = model_kg[trial_k]
                model = torch.load(model_path)

                try:
                    if batch_size == 1:
                        model_preds = predict(
                            model,
                            drug_labels[kg],
                            disease_labels[kg],
                            triples_factory
                        )
                    else:
                        model_preds = predict_mp(
                            model,
                            drug_labels[kg],
                            disease_labels[kg],
                            triples_factory
                        )
                except Exception as e:
                    print(f'Error using model {model_path}')
                    raise e

                model_preds['trial_k'] = trial_k
                model_preds['model'] = model_topo
                model_preds['kg'] = kg

                if save:
                    save_raw_predictions(model_preds, output_fn)

                all_models_predictions = pd.concat(
                    [all_models_predictions, model_preds],
                    ignore_index=True)

    return all_models_predictions
