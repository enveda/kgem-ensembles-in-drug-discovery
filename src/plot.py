# -*- coding: utf-8 -*-

"""Script for creating visualization plots."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import analysis
from src.constants import PLOTS_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_heatmap_models_intersection(df, figsize=(18, 18), save_to=None, title=None):
    unique_Ks = df['K'].unique()
    kgs = df['kg'].unique()
    fig, axes = plt.subplots(nrows=len(unique_Ks), ncols=len(kgs), figsize=figsize)

    for col_id, kg in enumerate(kgs):
        for row_id, K in enumerate(unique_Ks):
            intersection_pivot = df[
                (df['kg'] == kg) & \
                (df['K'] == K)].pivot("model1", "model2", "intersection")

            ax = axes[row_id, col_id]
            sns.heatmap(
                intersection_pivot,
                vmin=0,
                vmax=K,
                ax=ax,

            )
            ax.set_title(f"{kg}@K={K}", fontsize=14)
    plt.tight_layout()
    if title is not None:
        fig.suptitle(title, fontsize=20)
    if save_to is not None:
        plt.savefig(save_to)


def plot_heatmap_trials_intersection(df, models, save_to=None):
    unique_Ks = df['K'].unique()
    kgs = df['kg'].unique()

    for kg in kgs:
        fig, axes = plt.subplots(nrows=len(unique_Ks), ncols=len(models), figsize=(18, 18))
        for col_id, model in enumerate(models):
            for row_id, K in enumerate(unique_Ks):
                intersection_pivot = df[
                    (df['kg'] == kg) & \
                    (df['model'] == model) & \
                    (df['K'] == K)].pivot("trial1", "trial2", "intersection")

                ax = axes[row_id, col_id]
                sns.heatmap(
                    intersection_pivot,
                    #             robust=True,
                    vmin=0,
                    vmax=K,
                    ax=ax,

                )
                ax.set_title(f"{kg} - {model}@K={K}", fontsize=14)
        plt.tight_layout()
        if save_to is not None:
            plt.savefig(os.path.join(PLOTS_PATH, save_to))


def plot_precision_data(
    precision_data: pd.DataFrame,
    hue,
    title=None,
    hue_order=None,
    palette=None,
    nrows=2,
    ncols=1,
    figsize=(18, 18),
    save_to=None
):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    try:
        kgs = precision_data['kg'].unique()
    except Exception as e:
        print(precision_data.columns)
        raise e
    if hue_order is None:
        hue_order = precision_data[hue].unique()
    if palette is None:
        palette = sns.color_palette('tab10')

    legend = True

    for ax, kg in zip(axes.flatten(), kgs):
        df = precision_data[
            # (precision_data['trial_k'] == 0) & \
            (precision_data['kg'] == kg)
        ]

        g = sns.barplot(
            data=df,
            hue=hue,
            hue_order=hue_order,
            y="precision",
            x="K",
            palette=palette,
            ax=ax,
        )
        ax.set_title(f'{kg}', fontsize=18)
        if legend:
            legend = False
        else:
            ax.get_legend().remove()

    if title is not None:
        fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    if save_to is not None:
        plt.savefig(os.path.join(PLOTS_PATH, save_to))


def plot_score_distribution(
    df,
    title=None,
    nrows=2,
    ncols=1,
    figsize=(18, 18),
    omit_normalized=False,
    save_to=None
):
    models = df.model.unique()
    if omit_normalized:
        normalized_models = [m for m in df.model.unique() if analysis.is_normalized(df[df['model'] == m])]
        models = [m for m in df.model.unique() if m not in normalized_models]
        if len(normalized_models) > 0:
            logger.info(f"Omitting plot of models with normalized scores: {normalized_models}")

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax, kg in zip(axs, df.kg.unique()):
        sns.histplot(
            data=df[(df['kg'] == kg) & df['model'].isin(models)],
            x='score',
            hue='model',
            hue_order=models,
            ax=ax,
            # bins=30
        )
        ax.set_title(kg)

    if title is not None:
        fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    if save_to is not None:
        plt.savefig(os.path.join(PLOTS_PATH, save_to))
