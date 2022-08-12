# Ensembles of knowledge graph embedding models improve predictions for drug discovery

[comment]: <> ([![DOI:10.1016/j.ailsci.2022.100036]&#40;http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg&#41;]&#40;https://doi.org/10.1016/j.ailsci.2022.100036&#41;)

[comment]: <> ([![Arxiv]&#40;https://img.shields.io/badge/ArXiv-2105.10488-orange.svg&#41;]&#40;https://arxiv.org/abs/2105.10488&#41;)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pykeen)](https://img.shields.io/pypi/pyversions/pykeen)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[comment]: <> ([![License]&#40;https://img.shields.io/badge/License-Apache_2.0-blue.svg&#41;]&#40;https://opensource.org/licenses/Apache-2.0&#41;)

[comment]: <> (<p align="center">)

[comment]: <> (  <img width="800" src="https://github.com/AstraZeneca/kgem-in-drug-discovery/raw/master/result.png">)

[comment]: <> (</p>)

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
