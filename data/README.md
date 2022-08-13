## Directory structure

The current directory is structured in the following way:

```
|-- kg (Knowledge graph files for KGEMs)
|   |-- biokg
|   |   |-- test.tsv
|   |   |-- train.tsv
|   |   `-- val.tsv
|   `-- openbiolink
|       |-- test.tsv
|       |-- train.tsv
|       `-- val.tsv
|-- kgem-params (Model parameters)
|   |-- ComplEx.json
|   |-- ConvE.json
|   |-- DistMult.json
|   |-- ERMLP.json
|   |-- HolE.json
|   |-- MuRe.json
|   |-- RESCAL.json
|   |-- RotatE.json
|   |-- TransE.json
|   `-- TransH.json
|-- network (Raw knowledge graph files)
|   |-- biokg.links.tsv
|   `-- biokg_processed.tsv
`-- plots (Plots used in the manuscript)
    |-- ensemble_vs_agg_avg.png
    |-- ensembles_vs_best.png
    |-- graph-stats.png
    |-- grid_score_distribution_kgs.png
    |-- normalization_wrt_baseline.png
    |-- precision_boxplot_at10.png
    |-- precision_boxplot_at100.png
    |-- score_distribution_kgs-normalized-filtered.png
    |-- score_distribution_kgs-normalized.png
    |-- score_distribution_kgs.png
    |-- tp_vs_fp-heatmap-biokg.png
    |-- tp_vs_fp-heatmap-openbiolink.png
```
