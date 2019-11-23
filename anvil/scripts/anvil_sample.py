"""
anvil-sample

sample from trained models
"""
import logging

from reportengine.app import App
from anvil.config import ConfigParser

log = logging.getLogger(__name__)

PROVIDERS = ["anvil.sample", "anvil.observables", "anvil.plot", "anvil.checkpoint", "reportengine.report"]


class SampleApp(App):
    config_class = ConfigParser

    def __init__(self, name="validphys", *, providers):
        super().__init__(name, providers)


def main():
    a = SampleApp(name="anvil-sample", providers=PROVIDERS)
    a.main()


if __name__ == "__main__":
    main()
