# A Non-Volume preserving transformation Implementation on the Lattice - ANVIL

Framework for generating lattice proposals for a MCMC from a real NVP model

## Installation

At present installing the code into a conda environment is supported. First
create a new environment and install conda dependencies

```bash
conda create -n anvil-dev -y
conda install pytorch torchvision -c pytorch
conda install reportengine -c https://packages.nnpdf.science/conda
conda install tqdm, scipy
```

These are the minimal requirements for running the code, however if you plan
on developing the code or doing small external tests, then I highly recommend also
installing the following packages

```bash
conda install jupyter black pylint
```

To install the `anvil` code, run the following whilst in the root of this repository

```
$ python -m pip install -e .
```

where `python` here refers to the specific python of the conda environment with
the rest of the dependencies installed.

Once the code is installed you can import the various objects into your own
python projects. To get an idea of how to do this, look at the
`examples/train_example.py` which is discussed in
[this section](##using-objects-in-external-code.).

## Running tests

If you wish to run tests you will additionally need to install pytest

```bash
conda install pytest
```

the tests can be run from any location, provided the code is installed, with
the command

```
pytest --pyargs anvil
```

Consider adding tests whenever modifying/adding to the code.

## phi^4 example

To train a 2d model on the phi^4 action you will need a training runcard, an
example is given in `examples/runcards/l2_train_example.yml`. To train a new
model run

```
anvil-train <runcard name>.yml
```

by default this will create an output directory in your current directory
called `<runcard name>` however a specific output can also be given

```
anvil-train <runcard name>.yml -o <custom output name>
```

each new model requires its own output directory, however existing models can
be trained further by instead by giving an exisiting fit output directory as
input as well as telling the fit which epoch to start the retrain from

```
anvil-train <existing output name> -r 1000
```

this will recommence training at epoch 1000 using a checkpoint in the
`<existing output name>`, provided it exists. Note that you can just start from
the last checkpoint in the output directory by specifying `-r -1` which uses the
standard python syntax for index the final entry in, for example, a list.

Once you are satisfied that the training is finished you can generate plots,
tables and even reports. To see which observables have been implemented run
`anvil-sample --help anvil.observables`, feel free to open PRs implementing more!
An example on how to generate a simple report is given in
`examples/runcards/training_report.yml`. To use this runcard you will need a
training output with the name `l2_train_example` in your current working directory.
To generate the report simply run

```
anvil-sample training_report.yml
```

by default the outputted resources will be saved in `output` but you can specify
a custom output using the `-o` flag.

## Using objects in external code.

In the `examples` directory there is an example of using some of the objects
defined in models to learn a multigaussian distribution, you can run it with

```bash
$ ./train_example.py
```

which should train the model, and output the plots into `./example_output/`.
If the model successfully trains, then the ratio of the covariance matrix
sampled from the trained model over the input covariance should be approximately
1 as shown in the plot below

![ratio plot of sampled vs. target covariance](./examples/example_output/ratio.png)

if you have any issues using the anvil tools in your own projects feel free to
open an issue.
