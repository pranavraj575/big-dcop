# Dcop python

Tested with python 3.8 on Ubuntu 24

## Installation

(if using conda, create and activate environment)
```bash
conda create --name big-dcop python=3.8
conda activate big-dcop
```

Clone and install project
```bash
git clone https://github.com/pranavraj575/big-dcop
cd big-dcop
pip install -e .
```

Test installation

```bash
pydcop solve tests/instances/graph_coloring1.yaml --algo regret_matching --algo_param "stop_cycle:100"
```
## Run experiments

```bash
python evaluation/graph_coloring_generator.py
python evaluation/graph_coloring_runner.py 
python evaluation/plot_results.py 
```
## Formatting

To allow tests and formatting, install tuff:
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
 
