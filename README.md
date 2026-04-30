# Dcop python

Tested with python 3.8 on Ubuntu 24

## Installation

* (if using conda)
    ```bash
    conda create --name big-dcop python=3.8
    conda activate big-dcop
    ```

Clone and install project
```bash
git clone https://github.com/pranavraj575/big-dcop
cd big-dcop
pip install uv
uv pip install -e .
```

Test installation

```bash
pydcop solve tests/instances/graph_coloring1.yaml --algo regret_matching --algo_param "stop_cycle:100"
```

Note: Some plotting may require latex installation:

```shell
sudo apt install cm-super
sudo apt install dvipng
```
## Run experiments

### Graph coloring
To generate graphs, run experiments, then plot results:
```shell
python evaluation/graph_coloring_generator.py
python evaluation/graph_coloring_runner.py 
python evaluation/plot_results.py 
```
(run each script with `--help` to see options)

To create a visualization:
* Make a graph coloring instance 
  ```shell
  python evaluation/graph_coloring_generator.py --output_dir output/gif_graph_instances/ --num_problems 1 --graph_n 10
  ```
* Run `make_gif.py`:
  ```shell
  python evaluation/make_gif.py output/gif_graph_instances/gc_n10_k3_random_1.yaml --algorithms evaluation/configs/algorithm_configs.json --display_time --dpi 300 --seed 13
  ```
![](https://github.com/pranavraj575/big-dcop/blob/master/output/graph_color_gifs_readme/RM.gif)

## Formatting

To allow tests and formatting, install ruff:
```shell
pip install ruff
```

To format (and check for errors) before pushing:
```shell
ruff check; ruff format;
```

Since pyDCOP does not use the formatting enforced by ruff, we ignore checking format of files in pydcop folders.
To check a specific file anyway, specify this in the ruff command:
```shell
ruff check pydcop/algorithms/regret_matching.py; ruff format pydcop/algorithms/regret_matching.py;
```
## pyDCOP info
[![Documentation Status](https://readthedocs.org/projects/pydcop/badge/?version=latest)](http://pydcop.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/Orange-OpenSource/pyDcop.svg?branch=master)](https://travis-ci.org/Orange-OpenSource/pyDcop)

pyDCOP is a python library for Distributed Constraints Optimization.
It contains implementations of several standard DCOP algorithms (MaxSum, DSA,
DPOP, MGM, etc.) and allows you to develop your own algorithms.

pyDCOP runs on python >= 3.6.

Documentation is hosted on 
[ReadTheDoc](https://pydcop.readthedocs.io)
 
