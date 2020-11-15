# 2020MCM-ICM-Football-Team-Strategy-Analysis
Chaoran Cheng

Xinyi Wang

Boqiang Zhang

14th Feb. 2020

## Description

Project for 2020 MCM/ICM contest, Problem D. Analyses of performances of football teams.

File structures are as following:

```
root
│   2000170.pdf 				# the final paper
│   2020_ICM_Problem_D.pdf		# the problem description
│   LICENSE
│   README.md
│
├───figures 			# figures for visualization
│       analysis.png
│       Duel.png
│       filter.png
│       ...
│
├───Machine Learning	# Machine learning approach
│   ├───codes_large			# codes run on the original whole dataset
│   │       data_utils.py		# data utils
│   │       network.py			# main program
│   │       plot_utils.py		# plot utils
│   │       transformerModel.py	# network architecture
│   │
│   ├───codes_small			# does run on the local excerpt dataset
│   │       dataprocess.py		# pre-process data
│   │       data_utils.py		# data utils
│   │       network.py			# main program
│   │       plot_utils.py		# plot utils
│   │       README.txt			# dataset description
│   │       requirements.txt	# requirements
│   │       transformerModel.py	# network archtecture
│   │
│   └───results				# numerical results
│           connectivity.txt		# the extracted features
│           filter_len.txt
│           indices.txt
│           match_indices.txt
│           train.txt				# the training procedures
│           training_England_3.txt
│           training_France_2.txt
│           training_Germany_4.txt
│           training_Italy_1.txt
│           training_large.txt
│           training_small.txt
│           training_Spain_5.txt
│
└───Traditional		# Traditional Statistical approach
    ├───codes			# regression and hypothesis testing
    │       Model.m
    │       plot2.mlx
    │       plot_loss.mlx
    │       Possession.m
    │
    └───results			# numerical results
            all_stat.xlsx
            Apprendix2.xlsx
            direct_stats.csv
            Table&Apprendix.xlsx
            used_stat.csv
```

