"""
sample_api.py

This module contains the ``reportengine`` programmatic API, initialized with the
``anvil-sample`` providers, Config and Environment.

Example:
--------
In ``anvil/examples/runcards`` directory, run the training example:

    anvil-train train.yml

Then you can use the API to access anvil-sample objects in a python environment.
For example in a python shell:

>>> from anvil.api import API
>>> from anvil.api import API
>>> sample = API.configs(
...     training_output="./train",
...     cp_id=-1,
...     sample_size=10000,
...     thermalization=1000,
...     sample_interval=None
... )
configs: 100%|██████████████████████████| 10000/10000 [00:01<00:00, 9152.14it/s]
>>> sample.shape
torch.Size([10000, 36])

Of course this is not limited to use in a python shell, and can be used in
jupyter notebooks, scripts, tests etc.

The training class uses a config class which subclasses
``anvil.config.ConfigParser``, however most "intermediate" objects, required
for the training, can still be accessed:

>>> base = API.base_dist(base="gaussian", lattice_length=6, lattice_dimension=2)
>>> type(base)
<class 'anvil.distributions.Gaussian'>
>>> target = API.target_dist(
...     target="phi_four",
...     lattice_length=6,
...     lattice_dimension=2,
...     parameterisation="standard",
...     couplings={"m_sq": 4, "g": 0}
... )
>>> type(target)
<class 'anvil.distributions.PhiFourScalar'>

One can also abuse the API to monkey patch objects in, instead of obtaining
them from the resource builder. Although this functionality is largely untested,
so might produce strange results.

>>> from anvil.api import API
>>> fake_metro_sample = [[[0, 1, 2, 3, 4]]]
>>> API.configs(_metropolis_hastings=fake_metro_sample)
[0, 1, 2, 3, 4]

where ``fake_metro_sample`` has replaced the namespace object
``_metropolis_hastings`` and as a result, removes the dependencies normally
associated with that object (such has a model, sample size etc.).

"""
import logging

from reportengine import api
from reportengine.environment import Environment

from anvil.scripts.anvil_sample import PROVIDERS
from anvil.config import ConfigParser

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(PROVIDERS, ConfigParser, Environment)
