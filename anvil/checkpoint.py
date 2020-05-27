"""
checkpoint.py

Module for loading neural networks and checkpoints - ensuring a copy of model
is made so that we don't get unexpected results

"""
from copy import deepcopy

from reportengine import collect

from anvil.models import NeuralNetwork

def neural_network(
    size_half,
    i_affine,
    hidden_shape=(24,),
    activation="leaky_relu",
    final_activation=None,
    do_batch_norm=False,
):
    """Returns an instance of NeuralNetwork to be used in real NVP

    Parameters
    ----------
    size_half: int
        Number of nodes in the input and output layer of the network
    hidden_shape: list like
        List like specifying the number of nodes in the intermediate layers
    activation: (str, None)
        Key representing the activation function used for each layer
        except the final one.
    final_activation: (str, None)
        Key representing the activation function used on the final
        layer.
    do_batch_norm: bool
        Flag dictating whether batch normalisation should be performed
        before the activation function.
    name: str
        A label for the neural network, used for diagnostics.
        """
    return NeuralNetwork(
        size_in=size_half,
        size_out=size_half,
        hidden_shape=hidden_shape,
        activation=activation,
        final_activation=final_activation,
        do_batch_norm=do_batch_norm,
        name=f"s{i_affine}"
    )

s_networks = collect("neural_network", ("affine_layer_index", "s_network_spec",))
t_networks = collect("neural_network", ("affine_layer_index", "t_network_spec",))

def loaded_checkpoint(checkpoint):
    if checkpoint is None:
        return None
    cp_loaded = checkpoint.load()
    return cp_loaded


def train_range(loaded_checkpoint, epochs):
    if loaded_checkpoint is not None:
        cp_epoch = loaded_checkpoint["epoch"]
        train_range = (cp_epoch, cp_epoch + epochs)
    else:
        train_range = (0, epochs)
    return train_range


def loaded_model(loaded_checkpoint, model):
    new_model = deepcopy(model)  # need to copy model so we don't get weird results
    if loaded_checkpoint is not None:
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    return new_model


def current_loss(loaded_checkpoint):
    if loaded_checkpoint is None:
        return None
    return loaded_checkpoint["loss"]
