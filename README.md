# SDTR: Soft Decision Tree Regressor for Tabular Data

This repository is the supplementary code for for paper "SDTR: Soft Decision Tree Regressor for Tabular Data" ([link](https://ieeexplore.ieee.org/document/9393908)).

Some I/O schemes and data downloaders are taken from NODE(https://github.com/Qwicen/node). Much appreciated.

# Usage:
Please refer to lib/data.py and use 
```
fetch_{dataset_name}()
```
functions to download datasets.

Then, simply run:
```
python3 single_sdtr.py
```
Dataset and parameters can be modified in the python file `single_sdtr.py`. Note that it will automatically perform a hyper-parameter search.

Some hyper-parameters for SDTR model(DenseBlockSDTR):

* input_dim: the dimension of input data.
* layer_dim: how many trees are contained in a single layer.
* num_layers: how many (boosted)layers of trees.
* tree_dim: The output dimension of a tree. i.e. the dimension of 'weight vector' at the tree's leaves.
* max_features: As the boosting process continues, the input_dim for the following layers are growing rapidly. This hyper-param controls the max number of features that a tree can process (preventing OOM).

We also provide the unofficial evaluation scripts for gcForest `gcf.py`, tabnet `tabnet_tuning.py` and NODE `notebooks/epsilon_node_multigpu.ipynb`.


# Tips:
Some essential packages are probably not included in `requirements.txt`, especially the packages for gcForest and tabnet.