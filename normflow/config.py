"""
config.py

Module to parse runcards
"""
from pathlib import Path

import yaml
import torch
import torch.optim as optim

from normflow.models import NormalisingFlow
from normflow.observables import Observables, PhiFourAction
from normflow.geometry import Geometry2D


class ConfigError(Exception):
    pass


class ConfigParser:
    # __slots__ = 'config', 'model', 'geometry', 'train', 'sample', 'observables'

    def __init__(self, runcard):
        with open(runcard, "r") as f:
            self.config = yaml.safe_load(f)

    @staticmethod
    def check_key(spec: dict, key: str, typing, allow_none=False):
        """Check Key exists and is correct type, if so return value"""
        try:
            spec[key]
        except KeyError as e:
            raise ConfigError(f"You must specify {key} in the runcard") from e
        if allow_none and spec[key] is None:
            return None
        if not isinstance(spec[key], typing):
            raise ConfigError(
                f"{key} should be of type {typing}, not {type(spec[key])}"
            )
        return spec[key]

    def parse_network(self, net_spec):
        # TODO: add more functionality here
        hid_nodes = tuple(self.check_key(net_spec, "hidden_nodes", list))
        return {"hidden_nodes": hid_nodes}

    def parse_model(self, model_spec):
        n_affine = self.check_key(model_spec, "n_affine", int)
        network_spec = self.check_key(model_spec, "network", dict)
        network = self.parse_network(network_spec)
        checkpoint = self.check_key(model_spec, "checkpoint", str, allow_none=True)
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            if not checkpoint.is_file():
                raise ConfigError(f"{checkpoint} is not a file.")
        return dict(
            n_affine=n_affine, hid_shape=network["hidden_nodes"], checkpoint=checkpoint
        )

    def parse_geometry(self, geom_spec):
        length = self.check_key(geom_spec, "lattice_length", int)
        size = length ** 2
        # dim = self.check_key(geom_spec, 'dimension', int)
        # split = self.check_key(geom_spec, 'split', str)
        return dict(length=length, size_in=size)

    def parse_train(self, train_spec, loss, start):
        epochs = self.check_key(train_spec, "epochs", int)
        if start is not None:
            stop = start + epochs
        else:
            start = 0
            stop = epochs
        save = self.check_key(train_spec, "save_interval", int)
        n_batch = self.check_key(train_spec, "n_batch", int)
        # minimiser
        return dict(start=start, stop=stop, loss=loss, save_int=save, n_batch=n_batch)

    def parse_sample(self, sample_spec):
        chain_length = self.check_key(sample_spec, "chain_length", int)
        therm = self.check_key(sample_spec, "thermalisation", int)
        autocorr = self.check_key(sample_spec, "autocorrelation", int)
        obs_list = self.check_key(sample_spec, "observables", list, allow_none=True)
        observables = self.parse_observables(obs_list)
        return dict(
            chain_length=chain_length,
            thermalisation=therm,
            autocorr=autocorr,
            observables=observables,
        )

    def parse_observables(self, obs_list):
        for obs in obs_list:
            if not hasattr(Observables, obs):
                raise ConfigError(
                    f"The following observable hasn't been implemented: {obs}"
                )
        return obs_list

    def parse_action(self, action_spec):
        m_sq = self.check_key(action_spec, "m_sq", (float, int))
        lam = self.check_key(action_spec, "lambda", (float, int))
        return dict(m_sq=m_sq, lam=lam)

    def resolve(self):
        for key in self.config.keys():
            if not hasattr(self, f"parse_{key}"):
                raise ConfigError(f"Unexpected key in runcard: {key}")
        if "train" not in self.config.keys() and "sample" not in self.config.keys():
            raise ConfigError("Must specify at least one of `train` and `sample`")
        geom_dict = self.parse_geometry(self.config["geometry"])
        mod_dict = self.parse_model(self.config["model"])

        if mod_dict["checkpoint"] is not None:
            print(f"Loading model from checkpoint: {mod_dict['checkpoint']}")
            checkp = torch.load(mod_dict["checkpoint"])
            mod_state = checkp["model_state_dict"]
            opt_state = checkp["optimizer_state_dict"]
            loss = checkp["loss"]
            start = checkp["epoch"]
        else:
            mod_state, opt_state, loss, start = None, None, None, None

        model = NormalisingFlow(
            n_affine=mod_dict["n_affine"],
            size_in=geom_dict["size_in"],
            affine_hidden_shape=mod_dict["hid_shape"],
        )
        optimizer = optim.Adadelta(model.parameters())

        if mod_state is not None:
            model.load_state_dict(mod_state)
            optimizer.load_state_dict(opt_state)

        act_dict = self.parse_action(self.config["action"])
        if "train" in self.config.keys():
            train = self.parse_train(self.config["train"], loss=loss, start=start)
        else:
            train = None

        if "sample" in self.config.keys():
            if mod_dict["checkpoint"] is None and "train" not in self.config.keys():
                raise ConfigError(
                    "If sampling, either specify to `train`, or specify checkpoint in `model`"
                )
            sample = self.parse_sample(self.config["sample"])
        else:
            sample = None

        geometry = Geometry2D(geom_dict["length"])
        action = PhiFourAction(act_dict["m_sq"], act_dict["lam"], geometry)
        return model, optimizer, action, geometry, train, sample
