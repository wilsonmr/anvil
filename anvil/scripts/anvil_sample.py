# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
anvil-sample

sample from trained models
"""
import logging

from reportengine.app import App
from anvil.config import ConfigParser
from anvil.style import default_style

log = logging.getLogger(__name__)

PROVIDERS = [
    "anvil.models",
    "anvil.sample",
    "anvil.observables",
    "anvil.plot",
    "anvil.table",
    "anvil.checkpoint",
    "reportengine.report",
    "anvil.benchmarks",
]


class SampleApp(App):
    config_class = ConfigParser

    def __init__(self, name="anvil-sample", *, providers):
        super().__init__(name, providers)

        self.default_style = default_style


def main():
    a = SampleApp(name="anvil-sample", providers=PROVIDERS)
    a.main()


if __name__ == "__main__":
    main()
