.. _devinstall:

Development install
===================

If you wish to develop the code then follow the instructions in
:ref:`condainstall`. Then you can clone the
`git repo <https://github.com/wilsonmr/anvil>`__, following the instructions
on Github.

Navigate to the root of the git repo and then run the following in your ``anvil``
``conda`` environment to replace the ``anvil`` package you downloaded with
a development installation of the code:

.. code::

    $ python -m pip install -e .

Any changes you make in the git repo should be automatically reflected in
the ``anvil`` installation in your environment.

Developing and testing
----------------------

If you plan on developing the code, then we highly recommend also installing the following packages:

 - `Jupyter <https://jupyter.org/>`_
 - `black <https://pypi.org/project/black/>`_
 - `pylint <https://pypi.org/project/pylint/>`_
 - `pytest <https://pypi.org/project/pytest/>`_

Those packages can be installed using ``conda``:

.. code::

    conda install jupyter black pylint pytest

Where possible we try to follow the guidelines on formatting set out by
``black``, we also try to make good commit messages. We ask that if you
add any code, you document it properly and where appropriate add some tests.

We also ask that you ensure that none of the existing tests have failed as a
result of your changes. If you wish to run the tests, then the test dependencies
can be found in
``conda-recipe/meta.yaml`` under ``test::requires``. Simply install the
dependencies via ``conda``.

.. code::

    $ pytest --pyargs anvil

The core aim of this project is to provide a valuable resource for researching
physics and ML.
This is only true as long as our code is easy to understand and free
from bugs.

Finally, we also ask that you read our :ref:`codeofconduct`. We hope you enjoy
using our code!
