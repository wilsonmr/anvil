# -*- coding: utf-8 -*-
"""
norm_flow_pytorch.py

Module containing the base classes for affine layers and full normalising flow
model used to transform a simple distribution into an estimate of a target
distribution as per https://arxiv.org/abs/1904.12072

Key imports from this module are NormalisingFlow and shifted_kl. Which are the
necessary ingredients to train your own normalising flow model.
"""
from math import sqrt, pi

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_FUNC = F.leaky_relu # Globally define activation function

def probability_normal(x_tensor: torch.Tensor) -> torch.Tensor:
    """Caculate the probability of states, if each node was drawn from
    a normal distribution with mean 0, variance 1

    Parameters
    ----------
    x_tensor: torch.Tensor
        A 2-D tensor. Each row of x_tensor represents a state and each column
        represents a lattice site

    Returns
    -------
    out: torch.Tensor
        A 2-D tensor with a single column. Each entry is the probability of the
        corresponding state of the input `x_tensor`

    Example
    -------
    Consider the intergral of the domain [-1, 1]:

    >>> x = torch.linspace(-1, 1, 1000).view(-1, 1)
    >>> delta_x = 1/torch.shape[0]
    >>> p_x = probability_normal(x)
    >>> delta_x*torch.sum(p_x)*(1 - -1) # integrate
    tensor(0.6825)

    """
    exponent = -torch.sum(pow(x_tensor, 2)/2, dim=-1, keepdim=True)
    normalization = sqrt(pow(2*pi, x_tensor.shape[-1]))
    return torch.exp(exponent)/normalization

class AffineLayer(nn.Module):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    Affine transformation, z = g_i(\phi), defined as:

        z_a = \phi_a
        z_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    where D-dimensional phi has been split into two D/2-dimensional pieces
    \phi_a and \phi_b. s_i and t_i are neural networks, whose parameters are
    torch.nn parameters of this class.

    In the case of this class the partitions are expected to
    be the first and second part of the input data, an additional transformation
    will therefore need to be applied to the output/input of this network if
    the desired partition is to have a different pattern, for example a
    checkerboard transformation.

    Input and output data are flat vectors stacked in the first dimension
    (batch dimension).

    Parameters
    ----------
    size_in: int
        number of dimensions, D, of input/output data. Data should be fed to
        network in shape (N_states, size_in).
    s_hidden_shape: tuple
        tuple which gives the number of nodes in the hidden layers of neural
        network s_i, can be a single layer network with 16 nodes e.g (16,)
    t_hidden_shape: tuple
        tuple which gives the number of nodes in the hidden layers of neural
        network t_i.
    i_affine: int
        index of this affine layer in full set of affine transformations,
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.

    Attributes
    ----------
    s_layers: torch.nn.ModuleList
        the dense layers of network s, values are intialised as per the
        default initialisation of `nn.Linear`
    t_layers: torch.nn.ModuleList
        the dense layers of network t, values are intialised as per the
        default initialisation of `nn.Linear`

    Methods
    -------
    coupling_layer(phi_input):
        performs the transformation of a single coupling layer, denoted
        g_i(\phi)
    inverse_coupling_layer(z_input):
        performs the transformation of the inverse coupling layer, denoted
        g_i^{-1}(z)
    det_jacobian(phi_input):
        returns the contribution to the determinant of the jacobian
        = \frac{\partial g(\phi)}{\partial \phi}

    """
    def __init__(
            self,
            size_in: int,
            s_hidden_shape: tuple,
            t_hidden_shape: tuple,
            i_affine: int,
    ):
        super(AffineLayer, self).__init__()
        size_half = int(size_in/2)
        s_shape = [size_half, *s_hidden_shape, size_half]
        t_shape = [size_half, *t_hidden_shape, size_half]

        self.s_layers = nn.ModuleList(
            [
                nn.Linear(s_in, s_out)
                for s_in, s_out in
                zip(s_shape[:-1], s_shape[1:])
            ]
        )
        self.t_layers = nn.ModuleList(
            [
                nn.Linear(t_in, t_out)
                for t_in, t_out
                in zip(t_shape[:-1], t_shape[1:])
            ]
        )

        if (i_affine % 2) == 0: #starts at zero
            # a is first half of input vector
            self._a_ind = slice(0, int(size_half))
            self._b_ind = slice(int(size_half), size_in)
            self.join_func = torch.cat
        else:
            # a is second half of input vector
            self._a_ind = slice(int(size_half), size_in)
            self._b_ind = slice(0, int(size_half))
            self.join_func = lambda a, *args, **kwargs: torch.cat((a[1], a[0]), *args, **kwargs)

    def _s_forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Internal method which performs the forward pass of the network
        s.

        Input data x_input should be a torch tensor of size D, with the
        appropriate mask_mat applied such that elements corresponding to
        partition b are set to zero

        """
        for s_layer in self.s_layers:
            x_input = ACTIVATION_FUNC(s_layer(x_input))
        return x_input

    def _t_forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Internal method which performs the forward pass of the network
        t.

        Input data x_input should be a torch tensor of size D, with the
        appropriate mask_mat applied such that elements corresponding to
        partition b are set to zero

        """
        for t_layer in self.t_layers:
            x_input = ACTIVATION_FUNC(t_layer(x_input))
        return x_input

    def coupling_layer(self, phi_input) -> torch.Tensor:
        r"""performs the transformation of a single coupling layer, denoted
        g_i(\phi).

        Affine transformation, z = g_i(\phi), defined as:

        z_a = \phi_a
        z_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

        see eq. (9) of https://arxiv.org/pdf/1904.12072.pdf

        Parameters
        ----------
        phi_input: torch.Tensor
            stack of vectors \phi, shape (N_states, D)

        Returns
        -------
        out: torch.Tensor
            stack of transformed vectors z, with same shape as input

        """
        # since inputs are like (N_states, D) we need to index correct halves
        # of if input states
        phi_a = phi_input[:, self._a_ind]
        phi_b = phi_input[:, self._b_ind]
        s_out = self._s_forward(phi_a)
        t_out = self._t_forward(phi_a)
        z_b = (s_out.exp()*phi_b + t_out)
        return self.join_func([phi_a, z_b], dim=1) #put back together state

    def inverse_coupling_layer(self, z_input):
        r"""performs the transformation of the inverse coupling layer, denoted
        g_i^{-1}(z)

        inverse transformation, \phi = g_i^{-1}(z), defined as:

        \phi_a = z_a
        \phi_b = (z_b - t_i(z_a)) * exp(-s_i(z_a))

        see eq. (10) of https://arxiv.org/pdf/1904.12072.pdf

        Parameters
        ----------
        z_input: torch.Tensor
            stack of vectors z, shape (N_states, D)

        Returns
        -------
            out: torch.Tensor
                stack of transformed vectors phi, with same shape as input

        """
        z_a = z_input[:, self._a_ind]
        z_b = z_input[:, self._b_ind]
        s_out = self._s_forward(z_a)
        t_out = self._t_forward(z_a)
        phi_b = ((z_b - t_out)*torch.exp(-s_out))
        return self.join_func([z_a, phi_b], dim=1)

    def forward(self, phi_input):
        """Same as `coupling_layer`, but `forward` is a special method of a
        nn.Module which should be overridden
        """
        return self.coupling_layer(phi_input)

    def det_jacobian(self, phi_input):
        r"""returns the contribution to the determinant of the jacobian

            \frac{\partial g(\phi)}{\partial \phi} = prod_j exp(s_i(\phi)_j)

        see eq. (11) of https://arxiv.org/pdf/1904.12072.pdf

        Parameters
        ----------
        phi_input: torch.Tensor
            stack of vectors \phi, shape (N_states, D)

        Returns
        -------
        out: torch.Tensor
            column vector of contributions to jacobian (N_states, 1)

        """
        a_for_net = phi_input[:, self._a_ind] # select phi_a
        s_out = self._s_forward(a_for_net)
        return s_out.exp().prod(dim=-1)

class NormalisingFlow(nn.Module):
    r"""Extension to nn.Module which is built up of multiple `AffineLayer`s
    as per eq. (12) of https://arxiv.org/abs/1904.12072.

    Each affine layer transforms half of the input vector and the half of the
    input vector which is transformed is alternated. It is therefore recommended
    to have an even number of affine layers. Each affine layer has it's own
    pair of neural networks, s and t, which have seperate sets of parameters
    from other affine layers.

    For now each affine layer has networks with the same architecture
    for simplicity however this could be extended if required.

    Parameters
    ----------
    n_affine: int
        number of affine layers, it is recommended to choose an even number
    size_in: int
        number of nodes in the input/output vectors, data is expected to be
        given with shape (N_states, `size_in`)
    affine_hidden_shape: tuple
        tuple defining the number of nodes in the hidden layers of s and t.

    Attributes
    ----------
    affine_layers: torch.nn.ModuleList
        list of affine layers that form the full transformation

    """
    def __init__(
            self, *,
            n_affine: int = 2,
            size_in: int,
            affine_hidden_shape: tuple = (16,)
        ):
        super(NormalisingFlow, self).__init__()

        self.affine_layers = nn.ModuleList(
            [
                AffineLayer(size_in, affine_hidden_shape, affine_hidden_shape, i)
                for i in range(n_affine)
            ]
        )

    def map(self, x_input: torch.Tensor):
        """Function that maps field configuration to simple distribution"""
        raise NotImplementedError

    def inverse_map(self, z_input: torch.Tensor) -> torch.Tensor:
        r"""Function which maps simple distribution, z, to target distribution
        \phi.

        Parameters
        ----------
        z_input: torch.Tensor
            stack of simple distribution state vectors, shape (N_states, D)

        Returns
        -------
        out: torch.Tensor
            stack of transformed states, which are drawn from an approximation
            of the target distribution, same shape as input.

        """
        phi_out = z_input
        for layer in reversed(self.affine_layers): #reverse layers!
            phi_out = layer.inverse_coupling_layer(phi_out)
            # TODO: make this yield, then make a yield from wrapper?
        return phi_out

    def forward(self, phi_input: torch.Tensor) -> torch.Tensor:
        r"""Returns the log of the exact probability (of model) associated with
        each of the input states according to eq. (8) of
        https://arxiv.org/pdf/1904.12072.pdf.

        Parameters
        ----------
        phi_input: torch.Tensor
            stack of input states, shape (N_states, D)

        Returns
        -------
        out: torch.Tensor
            column of log(\tilde p) associated with each of input states

        """
        jacob_contr = torch.ones(phi_input.size()[0], 1)
        z_out = phi_input
        for layer in self.affine_layers:
            jacob_contr *= layer.det_jacobian(z_out).view(-1, 1)
            z_out = layer(z_out)
        simple_prob_z_out = probability_normal(z_out)
        return torch.log(simple_prob_z_out*jacob_contr)

def shifted_kl(log_tilde_p: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    r"""Sample mean of the shifted Kullbach-Leibler divergence between target
    and model distribution.

    Parameters
    ----------
    log_tilde_p: torch.Tensor
        column of log (\tilde p) for a sample of states, which is returned by
        forward pass of `NormalisingFlow` model
    action: torch.Tensor
        column of actions S(\phi) for set of sample states

    Returns
    -------
    out: torch.Tensor
        torch tensor with single element, corresponding to the estimation of the
        shifted K-L for set of sample states.

    """
    return torch.mean(log_tilde_p + action, dim=0)

if __name__ == "__main__":
    pass
