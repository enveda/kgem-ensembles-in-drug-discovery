# -*- coding: utf-8 -*-

"""Constants used through the code."""

import os

file_dir = os.path.dirname(os.path.realpath("__file__"))
root_dir = os.path.realpath(os.path.join(file_dir, '..'))

DATA_PATH = os.path.join(root_dir, 'data')
PREDICTIONS_PATH = os.path.join(DATA_PATH, 'predictions')
PLOTS_PATH = os.path.join(DATA_PATH, 'plots')
MODELS_PATH = os.path.join(DATA_PATH, 'models')
KGS_PATH = os.path.join(DATA_PATH, 'kg')

BENCHMARKS = {
    "cn": "Common Neighbors",
    "cos": "Cosine Similiarity",
    "ji": "Jaccard index",
    "si": "Sorensen index",
    "hpi": "Hub Promoted Index",
    "hdi": "Hub Depressed Index",
    "lhn": "Leicht–Holme–Newman Index",
    "pa": "Preferential Attachment",
    "aa": "Adamic-Adar",
    "ra": "Resource Allocation Index",
    "sp": "Shortest Path",
    "np": "Number of simple paths",
    "ug": "Number of unique genes in simple paths",
}

BENCHMARKS_LIST = list(BENCHMARKS.keys())

KGS = [
    "openbiolink",
    "biokg",
]

MODELS = [
    "mure", "transe", "rotate", "ermlp", "conve",
    "complex", "hole", "distmult", "transh"
]

DRUG_PREFIXES = {
    "openbiolink": "PUBCHEM",
    "biokg": "drugbank",
}
