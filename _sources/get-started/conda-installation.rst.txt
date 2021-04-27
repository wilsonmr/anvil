.. _condainstall:

Conda installation
==================

The supported method for installing the code and its dependencies is using
``conda``, a package manager supplied with either miniconda or anaconda.

We highly recommend setting up a separate ``conda`` environment for use with
anvil:

.. code::

    $ conda create -n anvil python-3.8

This will create a new environment with the supported version of ``python``.
Then installing the code and all of its dependencies can be achieved by simply
running

.. code::

    $ conda activate anvil # activate the environment
    $ conda install anvil -c wilsonmr -c pytorch -c https://packages.nnpdf.science/conda

Congratulations, you have installed ``anvil``.
