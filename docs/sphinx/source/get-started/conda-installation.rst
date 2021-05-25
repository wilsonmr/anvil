.. _condainstall:

Conda installation
==================

The supported method for installing the code and its dependencies is using
``conda``, a package manager supplied with either miniconda or anaconda. For
more information, check out
`<https://docs.conda.io/en/latest/>`_.

We highly recommend setting up a separate ``conda`` environment for use with
``anvil``:

.. code::

    $ conda create -n anvil python=3.8

This will create a new environment with the supported version of ``python``.
In order to install the code and all of its dependencies simply run:

.. code::

    $ conda activate anvil # activate the environment
    $ conda install anvil -c wilsonmr -c pytorch -c https://packages.nnpdf.science/conda

In the second line we specified that we wanted to search custom channels for
some of the packages (``anvil`` itself, ``pytorch`` and ``reportengine``).
Congratulations, you have installed ``anvil``.

If you wish to develop ``anvil``, then check out how to turn your installation
into a :ref:`devinstall`. Alternatively, if you're eager to start using
``anvil`` then check out some :ref:`basicusage`
