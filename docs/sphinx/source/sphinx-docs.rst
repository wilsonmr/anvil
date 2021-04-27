.. _sphinxdocs:

Building the documentation
==========================

In order to build the documentation, you will need ``sphinx``. You can install
this into the same ``conda`` environment as the code:

.. code::

    $ conda install sphinx

Once you have installed ``sphinx``, navigate to ``docs/sphinx`` and then run

.. code::

    $ make html

and the documentation should be built. The landing page for the docs will be
found (starting from the root of the git repo) at
``docs/sphinx/build/html/index.html``.
