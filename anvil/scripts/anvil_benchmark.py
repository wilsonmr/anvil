# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
anvil-benchmark

Train a simple free scalar theory model and check it against theoretical
predicitons, used to diagonose any problems in the anvil pipeline

"""
import logging
import shutil

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
    """Subclass the train app to add custom Config class"""
    config_class = BenchmarkTrainConfig


class BenchmarkSampleApp(anvil_sample.SampleApp):
    """Subclass the sample app to remove re-initialisation of logs."""
    def init_logging(self, args):
        pass # logging already initialised in training.


def main(_sample_runcard_path=None):
    """Main loop of benchmark"""
    train_app = BenchmarkTrainApp("benchmark-train", providers=anvil_train.PROVIDERS)
    sample_app = BenchmarkSampleApp(
        "benchmark-sample", providers=anvil_sample.PROVIDERS
    )
    train_app.main(
        cmdline=[
            str(benchmark_config.training_path.resolve()),
            "-o",
            BENCHMARK_OUTPUT
        ]
    )

    if _sample_runcard_path is not None:
        sample_rc_path = _sample_runcard_path
    else:
        sample_rc_path = benchmark_config.sample_path

    sample_app.main(
        cmdline=[
            str(sample_rc_path.resolve()),
            "-o",
            BENCHMARK_OUTPUT
        ]
    )


if __name__ == "__main__":
    main()
