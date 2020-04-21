"""
anvil-benchmark

Train a simple free scalar theory model and check it against theoretical
predicitons, used to diagonose any problems in the anvil pipeline

"""
import logging
import pathlib
import shutil

from reportengine.app import App

import anvil.scripts.anvil_train as anvil_train
import anvil.scripts.anvil_sample as anvil_sample
from anvil import benchmark_config

BENCHMARK_OUTPUT = "/tmp/del_me_anvil_benchmark"

log = logging.getLogger(__name__)


class BenchmarkTrainConfig(anvil_train.TrainConfig):
    """Update class for benchmarking"""
    @classmethod
    def from_yaml(cls, o, *args, **kwargs):
        if kwargs["environment"].output_path.is_dir():
            log.warning("deleting previous benchmark test results")
            shutil.rmtree(kwargs["environment"].output_path)
        return super().from_yaml(o, *args, **kwargs)


class BenchmarkTrainApp(anvil_train.TrainApp):
    config_class = BenchmarkTrainConfig
    """Subclass the train app to define our own settings"""
    def add_positional_arguments(self, parser):
        pass # intentionally don't set the config positional

    def get_commandline_arguments(self, cmdline=None):
        args = App.get_commandline_arguments(self, cmdline)
        args.update(dict(
            output=BENCHMARK_OUTPUT,
            config_yml=benchmark_config.training_path,
            retrain=None, # don't allow retraining
        ))
        return args

class BenchmarkSampleApp(anvil_sample.SampleApp):
    """Subclass the sample app to define out own settings"""
    def add_positional_arguments(self, parser):
        pass # see above

    def get_commandline_arguments(self, cmdline=None):
        args = super().get_commandline_arguments(cmdline)
        args.update(dict(
            output=BENCHMARK_OUTPUT,
            config_yml=benchmark_config.sample_path,
        ))
        return args


def main():
    """Main loop of benchmark"""
    train_app = BenchmarkTrainApp(
        "benchmark-train", providers=anvil_train.PROVIDERS)
    sample_app = BenchmarkSampleApp(
        "benchmark-sample", providers=anvil_sample.PROVIDERS)
    train_app.main()
    sample_app.main()

if __name__ == "__main__":
    main()
