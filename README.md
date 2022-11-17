<h1 align="center">
Ensembles of knowledge graph embedding models improve predictions for drug discovery
<br>

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://github.com/enveda/kgem-ensembles-in-drug-discovery/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7002695.svg)](https://doi.org/10.5281/zenodo.7002695)
![Maturity level-1](https://img.shields.io/badge/Maturity%20Level-ML--1-yellow)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pykeen)](https://img.shields.io/pypi/pyversions/pykeen)

</h1>

This repository accompanies the source code and data of the paper titled **"Ensembles of knowledge graph embedding models improve predictions for
drug discovery"**.

## Overview

In this work, we investigate the performance of ten knowledge graph embedding (KGE) models on two public biomedical
knowledge graphs (KGs). Till date, KGE models that yield higher precision on the top prioritized links are preferred.
In this paper, we take a different route and propose a novel concept of ensemble learning on KGEMs for drug discovery.
Thus, we assess whether combining the predictions of several models can lead to an overall improvement in predictive
performance. Our results highlight that such an ensemble learning method can indeed achieve better results than the
original KGEMs by benchmarking the precision (i.e., number of true positives prioritized) of their top predictions.

Below, the 10 models investigated in this paper: 

- [ComplEX](https://arxiv.org/abs/1606.06357)
- [ConvE](https://arxiv.org/abs/1707.01476)
- [DistMult](https://arxiv.org/abs/1412.6575)
- [ERMLP](https://dl.acm.org/doi/10.1145/2623330.2623623)
- [HolE](https://arxiv.org/abs/1510.04935)
- [MuRe](https://arxiv.org/abs/1905.09791)
- [RESCAL](http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf)
- [RotatE](https://arxiv.org/abs/1902.10197)
- [TransE](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [TransH](https://ojs.aaai.org/index.php/AAAI/article/view/8870)

With the following knowledge graphs benchmarked:

- [OpenBioLink](https://github.com/OpenBioLink/OpenBioLink)
- [BioKG](https://github.com/dsi-bdi/biokg)

The figure below provied the distribution of the Precision@100 achieved for each model trained with different hyperparameters in the OpenBioLink and BioKG KGs.

<p align="center">
  <img width="800" src="https://github.com/enveda/kgem-ensembles-in-drug-discovery/blob/main/data/plots/precision_boxplot_at100.png">
</p>


## Installation Dependencies

The dependencies required to run the notebooks can be installed as follows:

```shell
$ pip install -r requirements.txt
```

The code relies primarily on the [PyKEEN](https://github.com/pykeen/pykeen) package, which uses
[PyTorch](https://pytorch.org/) behind the scenes for gradient computation. If you want to train the models from scratch
it would be advisable to ensure you install a GPU enabled version of PyTorch first. Details on how to do this are
provided [here](https://pytorch.org/get-started/locally/).

## Reproducing Experiments 

This repository contains code to replicate the experiments detailed in the accompanying manuscript. Each model is
trained on a GPU server using the *train_model.py* script. 

Please note that the trained models will be saved in the **models** directory at the root of this repository within its
respective KG directory.

## Trained models and predictions
All the above mentioned models that were trained for the two KGs and their respecitve predictions can be found on [Zenodo](https://doi.org/10.5281/zenodo.7002695)

## Results and outcomes

We found that the baseline ensemble models outperformed each of the individual ones at all investigated K, highlighting the benefit of applying ensemble learning to KGEMs. The figure below shows the Precision at Top K in the test set using different values of K in the OpenBioLink and BioKG. For predefined values of K, the Precision@K for top predicted drug-disease triples are displayed for two ensembles (i.e., ensemble-all and ensemble-top5) and 2 independent KGEMs (i.e., RotatE and ConvE) using the 99th (BioKG) and 95th (OpenBioLink) percentile normalization approach. Although the latter two KGEMs represent the two best performing benchmarked models, the ensemble models outperform each of these individual models.

<p align="center">
  <img width="800" src="https://github.com/enveda/kgem-ensembles-in-drug-discovery/blob/main/data/plots/ensembles_vs_best.png">
</p>

## Repository structure

The current repository is structured in the following way:
```
|-- LICENSE
|-- README.md
|-- data (Data folder)
|   |-- kg
|   |   |-- biokg
|   |   `-- openbiolink
|   |-- kgem-params
|   |-- network
|   `-- plots
|-- notebooks (Python script for data processing)
|   |-- Step 1.0 - Data Pre-processing.ipynb
|   |-- Step 1.1 - Data Splitting.ipynb
|   |-- Step 2.1 - Score Distribution.ipynb
|   |-- Step 2.2 - KGEMs benchmarking.ipynb
|   |-- Step 2.3 - Validation-Test evaluation - Supplementary Table 1.ipynb
|   |-- Step 2.4 - Analyze Prediction Intersection.ipynb
|   |-- Step 3 - Exploration of Normalization methods.ipynb
|   `-- Step 3 - Analyze ensembles.ipynb
|-- requirements.txt
`-- src (Python utils for data manipulations)
    |-- analysis.py
    |-- constants.py
    |-- ensemble.py
    |-- full_pipeline.py
    |-- get_predictions.py
    |-- models.py
    |-- plot.py
    |-- predict.py
    |-- train_model.py
    |-- utils.py
    |-- version.py

```


## Citation
If you have found RPath useful in your work, please consider citing:

> Daniel Rivas-Barragan, Daniel Domingo-Fernández, Yojana Gadiya, David Healey, Ensembles of knowledge graph embedding models improve predictions for drug discovery, *Briefings in Bioinformatics*, 2022;, bbac481, [https://doi.org/10.1093/bib/bbac481](https://doi.org/10.1093/bib/bbac481)
