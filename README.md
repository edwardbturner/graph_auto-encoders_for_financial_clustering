# Repository for the paper "Graph Auto-Encoders for Financial Clustering"

## Requirements
```
Python 3.6
torch
torch_geometric
```
## Overview
This is a samll repository for my paper available at: https://arxiv.org/abs/2111.13519.

The ```5_fold_cross_validation.py``` file provides the code to split graph data into sets where k-fold cross validation can be carried out.

The ```model.py``` and ```train_validation_test.py``` files provide the graph auto-encoder used in the paper and the code to optimise it.

This is just the basic code used in the paper, if you want anything additional feel free to contact me.

If this is found to be helpful in your work consider refrencing the paper:

  @misc{turner2021graph,
      title={Graph Auto-Encoders for Financial Clustering}, 
      author={Edward Turner},
      year={2021},
      eprint={2111.13519},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
  }.
  
    $ [sudo] gem install jekyll-scholar

Or add it to your `Gemfile`:
