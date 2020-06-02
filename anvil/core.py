r"""
coupling.py
"""
import torch
import torch.nn as nn

from reportengine import collect

ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    None: nn.Identity,
}


class Sequential(nn.Sequential):
    """Modify the nn.Sequential class so that it takes an input vector *and* a
    value for the current logarithm of the model density, returning an output
    vector and the updated log density."""

    def forward(self, x_input, log_density):
        for module in self:
            x_input, log_density = module(x_input, log_density)
        return x_input, log_density


class NeuralNetwork(nn.Module):
    """Generic class for neural networks used in coupling layers.

    Networks consist of 'blocks' of
        - Dense (linear) layer
        - Batch normalisation layer
        - Activation function

    Parameters
    ----------
    size_in: int
        Number of nodes in the input layer
    size_out: int
        Number of nodes in the output layer
    hidden_shape: list
        List specifying the number of nodes in the intermediate layers
    activation: (str, None)
        Key representing the activation function used for each layer
        except the final one.
    final_activation: (str, None)
        Key representing the activation function used on the final
        layer.
    batch_normalise: bool
        Flag dictating whether batch normalisation should be performed
        before the activation function.
    label: str
        A label for the neural network, used for diagnostics.

    Methods
    -------
    forward:
        The forward pass of the network, mapping a batch of input vectors
        with 'size_in' nodes to a batch of output vectors of 'size_out'
        nodes.
    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        hidden_shape: list,
        activation: (str, None),
        final_activation: (str, None) = None,
        batch_normalise: bool = False,
        label: str = "network",
    ):
        super(NeuralNetwork, self).__init__()
        self.label = label
        self.size_in = size_in
        self.size_out = size_out
        self.hidden_shape = hidden_shape

        if batch_normalise:
            self.batch_norm = nn.BatchNorm1d
        else:
            self.batch_norm = nn.Identity

        self.activation_func = ACTIVATION_LAYERS[activation]
        self.final_activation_func = ACTIVATION_LAYERS[final_activation]

        # nn.Sequential object containing the network layers
        self.network = self._construct_network()

    def __str__(self):
        return f"Network: {self.label}\n------------\n{self.network}"

    def _block(self, f_in, f_out, activation_func):
        """Constructs a single 'dense block' which maps 'f_in' inputs to
        'f_out' output features. Returns a list with three elements:
            - Dense (linear) layer
            - Batch normalisation (or identity if this is switched off)
            - Activation function
        """
        return [
            nn.Linear(f_in, f_out),
            self.batch_norm(f_out),
            activation_func(),
        ]

    def _construct_network(self):
        """Constructs the neural network from multiple calls to _block.
        Returns a torch.nn.Sequential object which has the 'forward' method built in.
        """
        layers = self._block(self.size_in, self.hidden_shape[0], self.activation_func)
        for f_in, f_out in zip(self.hidden_shape[:-1], self.hidden_shape[1:]):
            layers += self._block(f_in, f_out, self.activation_func)
        layers += self._block(
            self.hidden_shape[-1], self.size_out, self.final_activation_func
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        """Forward pass of the network.
        
        Takes a tensor of shape (n_batch, size_in) and returns a new tensor of
        shape (n_batch, size_out)
        """
        return self.network(x)


class ConvexCombination(nn.Module):
    r"""
    Class which takes a set of normalising flows and constructs a convex combination
    of their outputs to produce a single output distribution and the logarithm of its
    volume element, calculated using the change of variables formula.

    A convex combination is a weighted sum of elements
        
        f(x_1, x_2, ..., x_N) = \sum_{i=1}^N \rho_i x_i

    where the weights are normalised, \sum_{i=1}^N \rho_i = 1.

    Parameters
    ----------
    flow_replica
        A list of replica normalising flow models.
    
    Methods
    -------
    forward(x_input, log_density):
        Returns the convex combination of probability densities output by the flow
        replica, along with the convex combination of logarithms of probability 
        densities.
    """

    def __init__(self, flow_replica):
        super().__init__()
        self.flows = nn.ModuleList(flow_replica)
        self.weights = nn.Parameter(torch.rand(len(flow_replica)))
        self.norm_func = nn.Softmax(dim=0)

    def forward(self, x_input, log_density):
        """Forward pass of the model.

        Parameters
        ----------
        x_input: torch.Tensor
            stack of input vectors drawn from the base distribution
        log_density: torch.Tensor
            The logarithm of the probability density of the base distribution.

        Returns
        -------
        out: torch.Tensor
            the convex combination of the output probability densities.
        log_density: torch.Tensor
            the logarithm of the probability density for the convex combination of
            output densities.
        """
        weights_norm = self.norm_func(self.weights)

        phi_out = 0
        for weight, flow in zip(weights_norm, self.flows):
            input_copy = (
                x_input.clone()
            )  # don't want each flow to update same input tensor
            zero_density = torch.zeros_like(
                log_density
            )  # don't want to add to base density
            phi_flow, log_dens_flow = flow(input_copy, zero_density)
            phi_out += weight * phi_flow
            log_density += weight * log_dens_flow

        return phi_out, log_density


_transformation_layers = collect(
    "transformation_layer", ("layer_indices", "layer_spec")  # must be this way round!
)

def normalising_flow(_transformation_layers, i_mixture=1):
    return Sequential(*_transformation_layers)


_flow_replica = collect("normalising_flow", ("mixture_indices",))


def convex_combination(_flow_replica):
    return ConvexCombination(_flow_replica)
