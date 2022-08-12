# -*- coding: utf-8 -*-

"""Script for getting top predictions based on models."""

import json
import os

import pandas as pd

from src.constants import KGS_PATH, MODELS_PATH, MODELS


def get_best_trial(model_dir: str):
    with open(f'{model_dir}/best_pipeline/pipeline_config.json', 'r') as f:
        best_trial = json.load(f)['metadata']['best_trial_number']
    return best_trial


def get_best_nth_trial(nth: int, model_dir: str):
    trials = pd.read_csv(f'{model_dir}/trials.tsv', sep='\t')
    trials = trials.sort_values('value', ascending=False)

    best_nth = trials.iloc[nth]['number']
    return best_nth


def get_topk_model_paths(top_k: int = 1):
    models_dir = MODELS_PATH
    model_paths = {}
    kgs = [
        kg for kg in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, kg))
    ]

    for kg in kgs:
        model_paths[kg] = {}
        model_kg_dir = f'{models_dir}/{kg}'
        model_topologies = [
            topo for topo in os.listdir(model_kg_dir)
            if os.path.isdir(os.path.join(models_dir, kg, topo)) and \
               topo in MODELS
        ]

        for model_topo in model_topologies:
            model_paths[kg][model_topo] = []
            for k in range(top_k):
                trained_model_path = f'{model_kg_dir}/{model_topo}/0000_user_data_{model_topo}'
                trial_to_load = get_best_nth_trial(k, trained_model_path)

                model_pkl = f'{trained_model_path}/artifacts/{trial_to_load}/trained_model.pkl'
                model_paths[kg][model_topo].append(model_pkl)

    return model_paths


def get_model_paths():
    models_dir = MODELS_PATH
    model_paths = {}
    kgs = [
        kg for kg in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, kg))
    ]

    for kg in kgs:
        model_paths[kg] = {}
        model_kg_dir = f'{models_dir}/{kg}'
        model_topologies = [
            topo for topo in os.listdir(model_kg_dir)
            if os.path.isdir(os.path.join(models_dir, kg, topo))
        ]
        for model_topo in model_topologies:
            trained_model_path = f'{model_kg_dir}/{model_topo}/0000_user_data_{model_topo}'
            best_pipeline = get_best_trial(trained_model_path)

            model_pkl = f'{trained_model_path}/artifacts/{best_pipeline}/trained_model.pkl'
            model_paths[kg][model_topo] = model_pkl

    return model_paths


def get_model_data():
    headers = ['source', 'relation', 'target']
    model_data_path = KGS_PATH

    columns = ['kg', 'dataset'] + headers
    model_data = pd.DataFrame([], columns=columns)
    for kg in os.listdir(model_data_path):
        for ds in ['train', 'test', 'val']:
            df = pd.read_csv(
                f'{model_data_path}/{kg}/{ds}.tsv',
                sep='\t', header=0, names=headers
            )
            df['kg'] = kg
            df['dataset'] = ds
            model_data = pd.concat([model_data, df], ignore_index=True)

    return model_data


def get_model_labels(model_data, dataset='test'):
    drug_labels = {}
    disease_labels = {}
    for kg in model_data['kg'].unique():
        drug_labels[kg] = model_data[
            (model_data['kg'] == kg) & \
            (model_data['dataset'] == dataset)]['source'].unique()
        disease_labels[kg] = model_data[
            (model_data['kg'] == kg) & \
            (model_data['dataset'] == dataset)]['target'].unique()

    return drug_labels, disease_labels
