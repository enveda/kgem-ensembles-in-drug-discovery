# -*- coding: utf-8 -*-

"""Script for running the prediction and visualization pipelines."""

import logging
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from src import analysis
from src import ensemble
from src import plot
from src import predict
from src import utils
from src.constants import MODELS, PLOTS_PATH, PREDICTIONS_PATH
from src.plot import plot_score_distribution

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUM_PROCS = 4


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess


def predict(topk_trials=1, datasets=['val', 'test']):
    for dataset in tqdm(datasets, desc="dataset"):
        _ = predict.get_all_predictions(
            MODELS,
            top_k_trials=topk_trials,
            save=True,
            skip_if_exists=True,
            return_all=True,
            dataset_to_predict=dataset,
        )


def load_predictions():
    predictions = pd.DataFrame([])

    glob = 'preds_*-trial_0-{}.csv'
    for dataset in ['val', 'test']:
        _glob = glob.format(dataset)
        predictions_ds = utils.merge_predictions(glob=_glob)
        predictions_ds['dataset'] = dataset
        predictions = pd.concat([predictions, predictions_ds], ignore_index=True)
    return predictions


def add_ground_truth(predictions):
    logger.info("Adding ground-truth column to predictions.")
    predictions_gt_path = os.path.join(PREDICTIONS_PATH, 'all_predictions-gt.csv')

    if os.path.exists(predictions_gt_path):
        predictions_gt = pd.read_csv(predictions_gt_path, sep='\t')
    else:
        predictions_gt = pd.DataFrame([])
        for dataset in ['val', 'test']:
            predictions_ds = predictions[predictions['dataset'] == dataset].copy().reset_index()
            predictions_ds = analysis.add_ground_truth(predictions_ds)
            predictions_ds['dataset'] = dataset
            predictions_gt = pd.concat([predictions_gt, predictions_ds], ignore_index=True)

        predictions_gt.to_csv(predictions_gt_path, sep='\t', index=False)
    return predictions_gt


@utils.timing
def get_predictions():
    predictions_path = os.path.join(PREDICTIONS_PATH, 'all_predictions.csv')
    predictions_gt_path = os.path.join(PREDICTIONS_PATH, 'all_predictions-gt.csv')

    predictions = None
    if os.path.exists(predictions_gt_path):
        logger.info("Reading csv with gt predictions.")
        predictions = pd.read_csv(predictions_gt_path, sep='\t')
    else:

        if os.path.exists(predictions_path):
            logger.info("Reading csv with predictions.")
            predictions = pd.read_csv(predictions_path, sep='\t')
        else:
            logger.info("Merging predictions from predictions' csv's.")
            predictions = load_predictions()
            predictions.to_csv(predictions_path, sep='\t', index=False)

        predictions = add_ground_truth(predictions)
    return predictions


@utils.timing
def get_precision(predictions, K_values, ensemble_id):
    precision_val_path = os.path.join(PREDICTIONS_PATH, f'precision_val-{ensemble_id}.csv')
    precision_test_path = os.path.join(PREDICTIONS_PATH, f'precision_test-{ensemble_id}.csv')

    precision_val = None
    if 'dataset' in predictions.columns and 'val' in predictions.dataset.unique():
        if os.path.exists(precision_val_path):
            logger.info("Reading validation precision from csv file.")
            precision_val = pd.read_csv(precision_val_path, sep='\t')
        else:
            logger.info("Computing validation precision.")
            precision_val = analysis.get_precision(
                predictions[predictions['dataset'] == 'val'],
                K_values=K_values,
                dataset='val',
                num_procs=NUM_PROCS,
            )
            precision_val.to_csv(precision_val_path, sep='\t', index=False)
    else:
        predictions['dataset'] = 'test'

    if os.path.exists(precision_test_path):
        logger.info("Reading test precision from csv file.")
        precision_test = pd.read_csv(precision_test_path, sep='\t')
    else:
        logger.info("Computing test precision.")
        precision_test = analysis.get_precision(
            predictions[predictions['dataset'] == 'test'],
            K_values=K_values,
            dataset='test',
            num_procs=NUM_PROCS,
        )
        precision_test.to_csv(precision_test_path, sep='\t', index=False)

    return precision_val, precision_test


@utils.timing
def get_models_intersection(predictions, K_values, ensemble_id):
    intersections_path = os.path.join(PREDICTIONS_PATH, f"{ensemble_id}_models_intersection.csv")
    if os.path.exists(intersections_path):
        models_intersection = pd.read_csv(intersections_path, sep='\t')
    else:
        models_intersection = analysis.get_models_intersection(predictions, K_values)
        models_intersection.to_csv(intersections_path, sep='\t', index=False)
    return models_intersection


def normalize_ensemble_df(
    df,
    norm_fn='scores',
    norm_topk=None,
):
    df_fn = pd.DataFrame([])
    for dataset in df.dataset.unique():
        for kg in df.kg.unique():
            for model in df.model.unique():
                for trial_k in df.trial_k.unique():
                    df_ = df[
                        (df['dataset'] == dataset) & \
                        (df['kg'] == kg) & \
                        (df['model'] == model) & \
                        (df['trial_k'] == trial_k)
                        ].copy().reset_index(drop=True)

                    if len(df_) == 0:
                        continue

                    try:
                        if norm_topk is not None:
                            if norm_topk >= 1:
                                topk_name = f'top{norm_topk}'
                                df_ = df_.head(norm_topk)
                            else:
                                percentile = norm_topk * 100
                                topk_name = f'perc{int(percentile)}'
                                min_score = np.percentile(df_['score'].values, percentile)
                                df_ = df_[df_['score'] >= min_score]

                        df_ = ensemble.apply_normalization(df_, norm_fn=norm_fn)
                        df_fn = pd.concat([df_fn, df_], ignore_index=True)
                    except Exception as e:
                        print(f'ERROR with model {model}')
                        logger.error(model)
                        print(f'min score: {min_score}')
                        print(f'len {len(df_)}')
                        raise e

    return df_fn


@utils.timing
def execute_pipeline(predictions, ensemble_id, ensemble_models):
    logger = mp.get_logger()
    process_id = mp.current_process()

    predictions = predictions[predictions['model'].isin(ensemble_models)].copy().reset_index()
    predictions_test = predictions[
        predictions['dataset'] == 'test'
        ]
    plot_score_distribution(
        df=predictions_test,
        title=f'{ensemble_id} models score distribution',
        save_to=f'{ensemble_id}-score_distribution_kgs.png',
    )

    plot_score_distribution(
        df=normalize_ensemble_df(predictions_test),
        title=f'{ensemble_id} models normalized score distribution',
        save_to=f'{ensemble_id}-score_distribution_kgs-normalized.png',
    )

    plot_score_distribution(
        df=normalize_ensemble_df(predictions_test, norm_topk=0.99),
        title=f'{ensemble_id} models Top percentile 99 normalized score distribution',
        save_to=f'{ensemble_id}-score_distribution_kgs-normalized-perc99.png',
    )

    # df_norm = utils.apply_fn(
    #     predictions_test,
    #     ensemble.apply_normalization,
    #     kwargs={'norm_fn': 'scores'})

    K_values = [1, 5, 10, 25, 50, 100, 250, 500]

    intersection_plot_path = os.path.join(PLOTS_PATH, f"{ensemble_id}-models_intersection-heatmap.png")
    if not os.path.exists(intersection_plot_path):
        logger.info(f"Computing models intersection on ensemble {ensemble_id}.")
        models_intersection = get_models_intersection(predictions_test, K_values, ensemble_id)
        plot.plot_heatmap_models_intersection(
            models_intersection,
            save_to=intersection_plot_path,
            title=f'Intersection in {ensemble_id}'
        )

    precision_val, precision_test = get_precision(predictions, K_values, ensemble_id)
    logger.info(f'{ensemble_id}: first precision')
    plot.plot_precision_data(
        precision_test,
        hue="model",
        title=f'{ensemble_id} models precision',
        save_to=f"{ensemble_id}_models_precision_atK.png"
    )

    if ensemble_id == "baseline":
        return [process_id, ensemble_id]

    norm_params = [
        ["scores", None],
        ["scores", 0.99],
        ["sigmoid-norm", None],
        ["sigmoid-avgcentered", None],
        ["sigmoid-norm", 0.99],
        ["sigmoid-avgcentered", 0.99],
    ]

    ensemble_precision = pd.DataFrame([])
    for norm_fn, norm_topk in norm_params:
        ensemble_precision_norm = ensemble.evaluate_ensemble(
            predictions_test,
            K_values=K_values,
            norm_fn=norm_fn,
            norm_topk=norm_topk,
            weights=None,
        )

        norm_topk_str = str(norm_topk) if norm_topk is not None else "all"
        logger.info(f'{ensemble_id}: second precision')
        plot.plot_precision_data(
            ensemble_precision_norm,
            title=f'{ensemble_id} precision with norm {norm_fn} for {norm_topk_str}',
            hue="model",
            save_to=f"{ensemble_id}_precision-{norm_fn}_{norm_topk_str}.png",
        )

        ensemble_precision = pd.concat([ensemble_precision, ensemble_precision_norm], ignore_index=True)

    palette1 = sns.color_palette("YlOrBr", n_colors=len(ensemble_models))
    palette2 = sns.color_palette("tab10", n_colors=len(ensemble_precision.model.unique()))

    logger.info(f'{ensemble_id}: third precision')
    plot.plot_precision_data(
        pd.concat([ensemble_precision, precision_test], ignore_index=True),
        hue="model",
        palette=palette1 + palette2,
        title=f'{ensemble_id} and its models\' precision',
        save_to=f"{ensemble_id}_vs_models_precision.png",
    )

    # Ensemble vs best model in ensemble
    best_precision = precision_test.groupby(['kg', 'K']).max().reset_index()
    best_precision['model'] = 'best'
    logger.info(f'{ensemble_id}: fourth precision')
    plot.plot_precision_data(
        pd.concat([ensemble_precision, best_precision], ignore_index=True),
        hue="model",
        title=f'Precision of {ensemble_id} vs best model in ensemble',
        save_to=f"{ensemble_id}_vs_best_precision.png",
    )

    score_norm_preds = ensemble.exploration_of_score_normalization(predictions_test)
    _, score_norm_precision = get_precision(score_norm_preds, K_values, f'{ensemble_id}-norm-exploration')

    print(score_norm_precision)
    score_norm_models = [m for m in score_norm_precision.model.unique() if m not in ensemble_models]
    palette2 = sns.color_palette("tab10", n_colors=len(score_norm_models))

    plot.plot_precision_data(
        pd.concat([precision_test, score_norm_precision], ignore_index=True),
        hue="model",
        hue_order=ensemble_models + score_norm_models,
        palette=palette1 + palette2,
        title=f'{ensemble_id} precision using different normalization functions',
        save_to=f"{ensemble_id}_norm_exploration_precision.png",
    )

    score_combination_precision = ensemble.exploration_of_score_combination(
        predictions,
        precision_data=precision_val,
        K_values=K_values,
    )

    score_combination_models = [m for m in score_combination_precision.model.unique() if m not in ensemble_models]
    palette2 = sns.color_palette("tab10", n_colors=len(score_combination_models))
    plot.plot_precision_data(
        score_combination_precision,
        hue="model",
        hue_order=ensemble_models + score_combination_models,
        palette=palette1 + palette2,
        title=f'{ensemble_id} precision using different combination methods',
        save_to=f"{ensemble_id}_combination_exploration_precision.png",
    )

    return [process_id, ensemble_id]


def execute_pipeline_mp(params):
    logger = mp.get_logger()

    process_id = mp.current_process()
    ensemble_id = params[1]
    logger.info(f"Process {process_id} evaluating {ensemble_id}")
    try:
        execute_pipeline(*params)
    except Exception as e:
        logger.error(e)
        raise e


def main():
    predictions = get_predictions()

    # tiny_path = os.path.join(PREDICTIONS_PATH, 'tiny_prediction.csv')
    # predictions = pd.read_csv(tiny_path, sep='\t')

    # tiny_df = pd.DataFrame([])
    # for model in predictions.model.unique():
    #     df = predictions[
    #         (predictions['dataset'] == 'test') & \
    #         (predictions['model'] == model)
    #     ].head(10).copy().reset_index()
    #     tiny_df = pd.concat([tiny_df, df], ignore_index=True)

    # tiny_df.to_csv(tiny_path, sep='\t', index=False)
    # return

    # predictions = pd.DataFrame([])

    # predictions_test = predictions[predictions['dataset'] == 'test']
    # plot_score_distribution(
    #     df=predictions_test,
    #     save_to='score_distribution_kgs.png',
    # )

    # K_values = [1, 5, 10, 25, 50, 100, 250, 500]

    # intersection_plot_path = os.path.join(PLOTS_PATH, "models_intersection-heatmap.png")
    # if not os.path.exists(intersection_plot_path):
    #     logger.info(f"Computing models intersection.")
    #     models_intersection = get_models_intersection(predictions_test, K_values)
    #     plot.plot_heatmap_models_intersection(
    #         models_intersection,
    #         save_to=intersection_plot_path,
    #     )

    # precision_val, precision_test = get_precision(predictions, K_values)
    # plot.plot_precision_data(
    #     precision_test,
    #     hue="model",
    #     save_to="baseline_precision_atK.png"
    # )

    # Ensembles:
    ensembles = {
        "ensemble-all": predictions['model'].unique(),
        "ensemble1": ['rotate', 'hole', 'transe', 'mure', 'complex'],
        "ensemble1-nocomplex": ['rotate', 'hole', 'transe', 'mure'],
        "ensemble2": ['transh', 'distmult', 'conve', 'rescal', 'ermlp'],
        "ensemble-top5": ['conve', 'transh', 'transe', 'mure', 'rotate'],
        "ensemble-bottom5": ['distmult', 'rescal', 'complex', 'hole', 'ermlp'],
        "ensemble-distance": ['mure', 'transe', 'transh', 'rotate'],
        "ensemble-semantic": ['distmul', 'rescal', 'complex', 'hole', 'ermlp', 'conve'],
    }

    pipeline_params_mp = [
        [predictions, ensemble_id, ensemble_models]
        for ensemble_id, ensemble_models in ensembles.items()
    ]

    # with MyPool(processes=8) as pool:
    pool = [
        mp.Process(
            target=execute_pipeline_mp,
            args=(params,)
        )
        for params in pipeline_params_mp]

    # for process, params in zip(pool, pipeline_params_mp):
    # results = process.
    for process_id, (proc, ensemble_id) in enumerate(zip(pool, ensembles.keys())):
        proc.start()
        logger.info(f"Process {process_id} started processing {ensemble_id}")

        # for process_id, (proc, ensemble_id) in enumerate(zip(pool, ensembles.keys())):
        proc.join()
        logger.info(f"Process {process_id} finished processing {ensemble_id}")

    # for process_id, ensemble_id in results:
    #     logger.info(f"Process {process_id} finished processing {ensemble_id}")
    logger.info("Finished evaluating all ensembles")


if __name__ == '__main__':
    main()
