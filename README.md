# Dcop python

Tested with python 3.8 on Ubuntu 24

## Installation

(if using conda, create and activate environment)
```bash
conda create --name dcop python=3.8
conda activate dcop
```

Clone and install project
```bash
git clone https://github.com/pranavraj575/big-dcop
cd big-dcop
pip install -e .
```

Test installation

```bash
pydcop -t 10 solve  --algo regret_matching tests/instances/graph_coloring1.yaml 
```
Run experiment

```bash
python evaluation/graph_coloring_generator.py
python evaluation/graph_coloring_runner.py 
```

# pyDCOP info
[![Documentation Status](https://readthedocs.org/projects/pydcop/badge/?version=latest)](http://pydcop.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/Orange-OpenSource/pyDcop.svg?branch=master)](https://travis-ci.org/Orange-OpenSource/pyDcop)

pyDCOP is a python library for Distributed Constraints Optimization.
It contains implementations of several standard DCOP algorithms (MaxSum, DSA,
DPOP, MGM, etc.) and allows you to develop your own algorithms.

pyDCOP runs on python >= 3.6.

Documentation is hosted on 
[ReadTheDoc](https://pydcop.readthedocs.io)
 
