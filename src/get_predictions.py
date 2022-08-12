# -*- coding: utf-8 -*-

"""Script for getting predictions from trained models."""

import argparse
import multiprocessing as mp

from src.constants import MODELS, KGS
from src.predict import get_all_predictions


def get_predictions_mp(model, kgs, topk_trials, datasets, skip_if_exists):
    for dataset in datasets:
        _ = get_all_predictions(
            model,
            kgs=kgs,
            top_k_trials=topk_trials,
            save=True,
            skip_if_exists=skip_if_exists,
            return_all=False,
            dataset_to_predict=dataset,
        )


def main():
    parser = argparse.ArgumentParser(description='Execute KGEM(s) and get predicted pairs.')
    parser.add_argument('-m', '--models', type=str, default=MODELS, nargs='+', help='List of KGEMs to execute')
    parser.add_argument('-t', '--topk-trials', type=int, default=1, help='Execute TopK trials per KGEM')
    parser.add_argument('-d', '--datasets', type=str, default=['val', 'test'], choices=['val', 'test'], nargs='+',
                        help='Datasets to use for each KG')
    parser.add_argument('-k', '--kgs', type=str, default=KGS, nargs='+', help='List of KGs to execute')
    parser.add_argument('--skip', default=False, action='store_true', help='Skip if predictions file exists')
    parser.add_argument('-p', '--parallel', default=False, action='store_true', help='Execute predictions in parallel.')

    args = parser.parse_args()

    params = [[[m], args.kgs, args.topk_trials, args.datasets, args.skip] for m in args.models]
    with mp.Pool(min(4, len(args.models))) as pool:
        results = pool.starmap(get_predictions_mp, params)
        for result in results:
            continue


if __name__ == '__main__':
    mp.set_start_method("spawn")
    main()
