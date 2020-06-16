"""
anvil-sample

sample from trained models
"""
import logging

from reportengine.app import App
from anvil.config import ConfigParser

log = logging.getLogger(__name__)

PROVIDERS = [
    "anvil.core",
    "anvil.models",
    "anvil.sample",
    "anvil.models",
    "anvil.observables",
    "anvil.plot",
    "anvil.table",
    "anvil.checkpoint",
    "reportengine.report",
    "anvil.benchmarks",
    "anvil.fields",
]


class SampleApp(App):
    config_class = ConfigParser

    def __init__(self, name="anvil-sample", *, providers):
        super().__init__(name, providers)


def main():
    a = SampleApp(name="anvil-sample", providers=PROVIDERS)
    a.main()


if __name__ == "__main__":
    main()
