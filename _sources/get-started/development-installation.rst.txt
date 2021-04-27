.. _devinstall:

Development install
===================

If you wish to develop the code then follow the instructions in
:ref:`condainstall`. Then you can clone the
`git repo <https://github.com/wilsonmr/anvil>`__, following the instructions
on Github.

Navigate to the root of the git repo and then run the following in the anvil
``conda`` environment to replace the ``anvil`` package you downloaded with
a development installation of the code:

.. code::

    $ python -m pip install -e .

Any changes you make in the git repo should be automatically reflected in
the ``anvil`` installation in your environment.
