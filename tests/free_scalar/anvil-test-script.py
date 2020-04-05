import logging

from reportengine.app import App
from anvil.config import ConfigParser

log = logging.getLogger(__name__)

PROVIDERS = ["anvil.sample", "anvil.checkpoint", "reportengine.report", "benchmarks"]

class TestApp(App):
    config_class = ConfigParser

    def __init__(self, name="validphys", *, providers):
        super().__init__(name, providers)

def main():
    a = TestApp(name="anvil-test-scalar", providers=PROVIDERS)
    a.main()

if __name__ == "__main__":
    main()
