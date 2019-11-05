# Git repo for flow-based generative model on lattice

Framework for generating lattice proposals for a MCMC from a real NVP model

## Installation

To install the code, run the following whilst in the root of this repository

```
$ python -m pip install -e .
```

this will give you a development installation of the code into the
same python environment as the python you used to run the command. That means
changes you make to the source files in normflow will be reflected without
having to reinstall the code.

you can then import the various objects into your own python projects. To get
an idea of how to do this, look at the `examples/train_example.py`


## phi^4 example

The script can be run either by training a new model, or by loading a model. To
train, run

```bash
$ normflowphifour train <model_name>.pt
```

or to load an exisitng model, run

```bash
$ normflowphifour load <model_name>.pt
```

The models will be saved to/loaded from a `models` directory. The output will be
a set of field configurations - each row of the output matrix corresponds to a
flattened configuration. The number of configurations generated is set by
`target_length` in the `main` method.


## Toy example

In order to run the toy example you will need a python environment with `python`
3.7, `pytorch`, `tqdm` and some of the other common 3rd party modules like
`numpy` and `matplotlib`.

Once you have this, go to the `examples` directory and run

```bash
$ ./train_example.py
```

which should train the model, and output the plots into `./example_output/`.
If the model successfully trains, then the ratio of the covariance matrix
sampled from the trained model over the input covariance should be approximately
1

![ratio plot of sampled vs. target covariance](./examples/example_output/ratio.png)
