# -*- coding: utf-8 -*-

"""Utils for model."""

import pathlib
from functools import wraps
from time import time

import networkx as nx
import pandas as pd
import torch
from pykeen.models import Model

from src.constants import PREDICTIONS_PATH


def create_graph_from_df(
    graph_df
) -> nx.DiGraph:
    """Create fully connected graph from dataframe."""
    graph = nx.DiGraph()

    for sub_name, obj_name, relation in graph_df.values:
        # Store edge in the graph
        graph.add_edge(
            sub_name,
            obj_name,
            polarity=relation,
        )

    connected_components_subgraph = [
        component
        for component in sorted(
            nx.connected_components(
                graph.to_undirected()
            ),
            key=len,
            reverse=True
        )
    ]

    final_subgraph = graph.subgraph(connected_components_subgraph[0])

    return final_subgraph


def get_path_score(
    source: str,
    target: str,
    graph_df: pd.DataFrame,
    model: Model
):
    graph = create_graph_from_df(graph_df).to_undirected()
    shortest_paths = nx.all_shortest_paths(graph, source, target)

    all_scores = []

    for i, path in enumerate(shortest_paths):
        score_dict = {}
        for idx, node in enumerate(path[:-1]):
            source = path[idx]
            target = path[idx + 1]

            # Get source, target, relation data from model
            source_id = model.triples_factory.entities_to_ids([source])[0]
            target_id = model.triples_factory.entities_to_ids([target])[0]
            polarity = str(graph.get_edge_data(source, target)['polarity'])
            relation_id = model.triples_factory.relations_to_ids([polarity])[0]

            # Get distance between source and target node
            s = torch.LongTensor([[source_id, relation_id, target_id]])
            score = model.score_hrt(hrt_batch=s)
            score_dict[f'{source}_{target}'] = score
        all_scores.append(score_dict)

    return all_scores


def merge_predictions(glob, topk=None):
    csv_files = pathlib.Path(PREDICTIONS_PATH).glob(glob)
    predictions = pd.DataFrame([])
    for csv in csv_files:
        df = pd.read_csv(csv, sep='\t').sort_values('score', ascending=False)
        if topk is not None:
            df = df.head(topk).reset_index()

        predictions = pd.concat([predictions, df], ignore_index=True)

    predictions = predictions.drop('tail_id', axis=1)
    predictions = predictions.rename(columns={
        "head_label": "source",
        "tail_label": "target",
    })
    return predictions


def apply_fn(df, fn, **kwargs):
    df_fn = pd.DataFrame([])
    for dataset in df.dataset.unique():
        for kg in df.kg.unique():
            for model in df.model.unique():
                for trial_k in df.trail_k.unique():
                    df_ = df[
                        (df['dataset'] == dataset) & \
                        (df['kg'] == kg) & \
                        (df['model'] == model) & \
                        (df['trial_k'] == trial_k)
                        ]

                    df_ = fn(df_, **kwargs)
                    df_fn = pd.concat([df_fn, df_], ignore_index=True)

    return df_fn


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func: {f.__name__} took: {te - ts:.4f} sec')
        return result

    return wrap
