# Repository for the paper "Graph Auto-Encoders for Financial Clustering"

## Requirements
```
Python 3.6
torch
torch_geometric
```
## Overview
This is a samll repository for my paper available at: https://arxiv.org/abs/2111.13519.

The ```5_fold_cross_validation.py``` provides the code to perform k-fold cross validation 


This is a simple code to transform a graph (via it's adjacency matrix) from an excel sheet to a PyTorch Geometric graph object. The graph may be weighted, directed or have self loops, if you don't want any of these then simply delete that part of the spreadsheet (i.e. delete all 'feature x' rows). I created this as there seems to be a real lack of literature on creating your own PyTorch Geometric graphs online.

The ```excel_example.xls``` file is included to show the necessary excel layout, simply download and edit it to fit your graph. Note for the adjacency matrix of a weighted graph the (i, j) entry is the edge from node i to node j.

If this is found to be helpful in your work consider refrencing/linking the repository.
