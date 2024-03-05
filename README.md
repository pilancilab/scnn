# `scnn`: Scalable Convex Neural Networks 

A library for global optimization of shallow neural networks.
API documentation at [ReadTheDocs](https://scnn.readthedocs.io/en/latest/).

### Requirements

Python 3.8 or newer. Development dependencies are listed in `dev_requirements.txt`. 

### Setup

Install using `pip`:

```
python -m pip install pyscnn
```

Or, clone the repository and manually install: 

```
git clone https://github.com/pilancilab/scnn.git
python -m pip install ./scnn
```

### Contributions

Coming soon!

### Citation

Please cite our paper if you use this package.

```
@article{DBLP:journals/corr/abs-2202-01331,
  author    = {Aaron Mishkin and
               Arda Sahiner and
               Mert Pilanci},
  title     = {Fast Convex Optimization for Two-Layer ReLU Networks: Equivalent Model
               Classes and Cone Decompositions},
  journal   = {CoRR},
  volume    = {abs/2202.01331},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.01331},
}
```

Looking for the code to replicate our experiments?
See [scnn_experiments](https://github.com/aaronpmishkin/scnn_experiments).
