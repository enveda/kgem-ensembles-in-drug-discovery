# -*- coding: utf-8 -*-

"""Script to run the train model for link prediction.

Ex. >> run_pykeen(
        train_file_path='../data/kg/train.tsv',
        test_file_path='../data/kg/test.tsv',
        val_file_path='../data/kg/val.tsv',
        output_directory='../data/models',
        configuration='MuRe'
    )
"""

import json
import logging
import os

import pandas as pd
from pykeen.hpo.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory

dir_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)))

logger = logging.getLogger("__name__")


def run_pykeen(
    train_file_path: str,
    test_file_path: str,
    val_file_path: str,
    output_directory: str,
    configuration: str = "TransE",
):
    """Embed the given graph using the model and used for link prediction.
    :param train_file_path: directory to TSV for training triples
    :param test_file_path: directory to TSV for testing triples
    :param val_file_path: directory to TSV for validation triples
    :param output_directory: directory for where to store results and pickle of model
    :param configuration: the model name that you want to use for embedding
    :return hpo_pipeline that holds results for model
    """
    with open(
        os.path.join(PROJECT_DIR, "data", "kgem-params", f"{configuration}.json"), "r"
    ) as config_file:
        model_config = json.load(config_file)

    logger.info(f"running {configuration} model")

    # Load the data sets
    train_df = pd.read_csv(train_file_path, sep="\t").to_numpy()
    test_df = pd.read_csv(test_file_path, sep="\t").to_numpy()
    val_df = pd.read_csv(val_file_path, sep="\t").to_numpy()

    train = TriplesFactory.from_labeled_triples(train_df)
    test = TriplesFactory.from_labeled_triples(test_df)
    val = TriplesFactory.from_labeled_triples(val_df)

    results = hpo_pipeline(
        dataset=None,
        training=train,
        testing=test,
        validation=val,
        model=model_config["pipeline"]["model"],
        model_kwargs=model_config["pipeline"]["model_kwargs"],
        model_kwargs_ranges=model_config["pipeline"].get("model_kwargs_ranges"),
        loss=model_config["pipeline"]["loss"],
        loss_kwargs=model_config["pipeline"].get("loss_kwargs"),
        loss_kwargs_ranges=model_config["pipeline"].get("loss_kwargs_ranges"),
        # regularizer=model_config["pipeline"].get("regularizer"),
        optimizer=model_config["pipeline"]["optimizer"],
        optimizer_kwargs=model_config["pipeline"].get("optimizer_kwargs"),
        optimizer_kwargs_ranges=model_config["pipeline"].get("optimizer_kwargs_ranges"),
        training_loop=model_config["pipeline"]["training_loop"],
        training_kwargs=model_config["pipeline"].get("training_kwargs"),
        training_kwargs_ranges=model_config["pipeline"].get("training_kwargs_ranges"),
        negative_sampler=model_config["pipeline"].get("negative_sampler"),
        negative_sampler_kwargs=model_config["pipeline"].get("negative_sampler_kwargs"),
        negative_sampler_kwargs_ranges=model_config["pipeline"].get(
            "negative_sampler_kwargs_ranges"
        ),
        stopper=model_config["pipeline"].get("stopper"),
        stopper_kwargs=model_config["pipeline"].get("stopper_kwargs"),
        evaluator=model_config["pipeline"].get("evaluator"),
        evaluator_kwargs=model_config["pipeline"].get("evaluator_kwargs"),
        evaluation_kwargs=model_config["pipeline"].get("evaluation_kwargs"),
        n_trials=model_config["optuna"]["n_trials"],
        timeout=model_config["optuna"]["timeout"],
        metric=model_config["optuna"]["metric"],
        direction=model_config["optuna"]["direction"],
        sampler=model_config["optuna"]["sampler"],
        pruner=model_config["optuna"]["pruner"],
        save_model_directory=output_directory,
    )

    os.makedirs(output_directory, exist_ok=True)
    results.save_to_directory(output_directory)
    return results


##### Example on how to run the model #####
##### Modify here the model you want #####
# if "__main__" == __name__:
#     run_pykeen(
#         train_file_path="./data/kg/biokg/train.tsv",
#         test_file_path="./data/kg/biokg/test.tsv",
#         val_file_path="./data/kg/biokg/val.tsv",
#         output_directory="./data/",
#         configuration="ConvE",
#     )
