from pathlib import Path
import shutil
import logging
import re

from reportengine.app import App
from reportengine.environment import Environment
from reportengine.compat import yaml
from anvil.config import ConfigParser, ConfigError

log = logging.getLogger(__name__)

PROVIDERS = ["anvil.train", "anvil.checkpoint", "anvil.models"]

TRAINING_ACTIONS = ["train"]

RUNCARD_COPY_FILENAME = "runcard.yml"

INPUT_FOLDER_NAME = "input"

class TrainError(Exception):
    pass


class TrainConfig(ConfigParser):
    """Specialization for yaml parsing"""

    @classmethod
    def from_yaml(cls, o, *args, **kwargs):
        if (
            kwargs["environment"].output_path.is_dir()
            and kwargs["environment"].extra_args["retrain"] is None
        ):
            raise ConfigError(
                f"output directory: {kwargs['environment'].output_path} already exists, did you mean to retrain?"
            )
        try:
            file_content = yaml.safe_load(o)
        except yaml.error.YAMLError as e:
            raise ConfigError(f"Failed to parse yaml file: {e}")
        if not isinstance(file_content, dict):
            raise ConfigError(
                f"Expecting input runcard to be a mapping, "
                f"not '{type(file_content)}'."
            )
        extra_input = {
            "training_output": str(kwargs["environment"].output_path),
            "cp_id": kwargs["environment"].extra_args["retrain"],
            "outpath": str(kwargs["environment"].output_path / "checkpoints"),
            "actions_": ["train"],
        }
        for key, value in extra_input.items():
            file_content.setdefault(key, value)
        return cls(file_content, *args, **kwargs)


class TrainEnv(Environment):
    """Container for information to be filled at run time"""

    def init_output(self):

        if not self.config_yml.exists():
            raise TrainError("Invalid runcard. File not found.")

        if not self.config_yml.is_file():
            raise TrainError("Invalid runcard. Must be a file.")

        self.output_path = Path(self.output_path).absolute()

        if not re.fullmatch(r"[\w.\-]+", self.output_path.name):
            raise TrainError("Invalid output folder name. Must be alphanumeric.")

        # if retraining, output directories exist
        if self.extra_args["retrain"] is not None:
            return

        # make output directories and copy runcard
        self.output_path.mkdir()
        (self.output_path / "checkpoints").mkdir()
        (self.output_path / "logs").mkdir()
        self.input_folder = self.output_path / INPUT_FOLDER_NAME
        self.input_folder.mkdir()

        shutil.copy2(self.config_yml, self.output_path / RUNCARD_COPY_FILENAME)


class TrainApp(App):
    config_class = TrainConfig
    environment_class = TrainEnv

    def __init__(self, name="validphys", *, providers):
        super().__init__(name, providers)

    @property
    def argparser(self):
        parser = super().argparser
        output_or_retrain = parser.add_mutually_exclusive_group()
        output_or_retrain.add_argument(
            "-o", "--output", help="Output folder and name of the fit", default=None
        )
        output_or_retrain.add_argument(
            "-r",
            "--retrain",
            help="epoch from which to retrain from",
            type=int,
            default=None,
        )
        return parser

    def get_commandline_arguments(self, cmdline=None):
        args = super().get_commandline_arguments(cmdline)
        if args["retrain"] is not None:
            args["output"] = args["config_yml"]
            args["config_yml"] = args["config_yml"] + f"/{RUNCARD_COPY_FILENAME}"
        elif args["output"] is None:
            args["output"] = Path(args["config_yml"]).stem
        return args

    def add_positional_arguments(self, parser):
        parser.add_argument(
            "config_yml",
            help="path to the configuration file or existing output folder if specifying `retrain` epoch",
        )

    def run(self):
        self.environment.config_yml = Path(self.args["config_yml"]).absolute()
        super().run()


def main():
    a = TrainApp(name="anvil-train", providers=PROVIDERS)
    a.main()


if __name__ == "__main__":
    main()
