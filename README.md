# Repository for the paper "Graph Auto-Encoders for Financial Clustering"

## Requirements
```
Python 3.6
torch
torch_geometric
```
## Overview
This is a samll repository for my paper available at: https://arxiv.org/abs/2111.13519.

The ```5_fold_cross_validation.py``` provides the code to split graph data into sets where k-fold cross validation can be carried out.


This is a simple code to transform a graph (via it's adjacency matrix) from an excel sheet to a PyTorch Geometric graph object. The graph may be weighted, directed or have self loops, if you don't want any of these then simply delete that part of the spreadsheet (i.e. delete all 'feature x' rows). I created this as there seems to be a real lack of literature on creating your own PyTorch Geometric graphs online.

This is just the basic code used in the paper, if you want anything additional feel free to contact me.

If this is found to be helpful in your work consider refrencing the paper with:
  @misc{turner2021graph,
      title={Graph Auto-Encoders for Financial Clustering}, 
      author={Edward Turner},
      year={2021},
      eprint={2111.13519},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
  }.
