import numpy as np
from math import pi
from sys import maxsize

from anvil.utils import Multiprocessing, bootstrap_sample, spher_to_eucl, unit_norm

from reportengine import collect


class ScalarField:
    """
    Class for ensembles of scalar fields.

    Parameters
    ----------
    input_coords: numpy.ndarray
        The coordinates of the field ensemble.
        Valid inputs may have dimensions:
            - (lattice_volume, ensemble_size)
            - (lattice_volume, *, ensemble_size)
        The final dimension will always be treated as the ensemble dimension in the
        context of calculating ensemble averages.
    lattice: anvil.geometry.Geometry
        Lattice object upon which the fields are defined. Must have the same number of
        sites as the 0th dimension of `input_coords`.

    Notes
    -----
    (1) The lattice field theory is assumed to respect the lattice analogues of homogeneity
    and isotropy, i.e. to be invariant under global translations along the natural axes of
    the lattice, and under discrete rotations mapping lattice axes onto each other. This
    allows the definition of a two point correlation function that is a function of separations
    only (not absolute coordinates) by averaging over the lattice sites.
    """

    minimum_dimensions = 2

    def __init__(self, input_coords, lattice):

        self.lattice = lattice  # must do this before setting coords!

        # HACK
        self.coords = input_coords[0].transpose(0, 1).numpy()
        self.tau_chain = float(input_coords[1])
        self.acceptance = float(input_coords[2])

        self.shift = self.lattice.get_shift().transpose(
            0, 1
        )  # dimensions swapped wrt training

    def _valid_ensemble(self, array_in):
        # Check that 0th dimension of input data matches the number of lattice sites
        assert (
            array_in.shape[0] == self.lattice.volume
        ), f"Size of coordinates array at dimension 0: {array_in.shape[0]} does not match volume of lattice: {self.lattice.volume}"
        # Check that the input data has at least the minimum number of dimensions
        n_dims = len(array_in.shape)
        assert (
            n_dims >= self.minimum_dimensions
        ), f"Invalid number of dimensions in coordinate array. Expected {self.minimum_dimensions} or more, but found {n_dims}."
        return array_in

    @property
    def coords(self):
        """The set of coordinates which define the ensemble.
        numpy.ndarray, dimensions (lattice_volume, *, ensemble_size)"""
        return self._coords

    @coords.setter
    def coords(self, new):
        """Setter for coords which performs some basic checks (see _valid_field)."""
        self._coords = self._valid_ensemble(new)
        self._coords_pos = np.copy(self._coords)

    @property
    def magnetisation_series(self):
        return self.coords.sum(axis=0)

    @property
    def magnetisation(self):
        return magnetisation_series.mean(axis=-1)

    def _vol_avg_two_point_correlator(self, shift):
        """Helper function which calculates the volume-averaged two point correlation
        function for a single shift, given in lexicographical form as an argument.
        """
        return (self.coords[shift] * self.coords).mean(axis=0)

    @property
    def two_point_correlator_series(self):
        """The volume-averaged two point correlation function for the first few shifts
        along a single axis.
        numpy.ndarray, dimensions (n_shifts, *, ensemble_size)"""
        n_shifts = min(4, self.lattice.length // 2 + 1)
        correlator = []
        for i in range(n_shifts):
            shift = self.lattice.get_shift(((0, i),), ((0, 1),)).flatten()
            correlator.append(self._vol_avg_two_point_correlator(shift))
        return np.array(correlator)

    def _two_point_correlator(self, connected=False):
        """Helper function which calculates the two point correlation function for all
        shifts, using multiprocessing. The order of the sums over volume and ensemble
        are swapped for efficiency reasons - we can bootstrap the volume-averaged
        correlator rather than the ensemble of fields themselves."""
        # Compute first term, using multiprocessing
        mp_correlator = Multiprocessing(
            func=lambda shift: self._vol_avg_two_point_correlator(shift).mean(axis=-1),
            generator=self.lattice.two_point_iterator,
        )
        correlator_dict = mp_correlator()  # multiprocessing returns a dict
        extra_dims = correlator_dict[0].shape  # concurrent samples dimension

        # Expand dict to numpy array, dimensions (lattice.length**2, *extra_dims)
        correlator = np.array([correlator_dict[i] for i in range(self.lattice.volume)])

        if connected:  # subtract disconnected part: <|m|>^2
            correlator -= np.abs(self.coords.mean(axis=0)).mean(axis=-1) ** 2

        return correlator.reshape(
            (self.lattice.length, self.lattice.length, *extra_dims)
        )

    def _boot_two_point_correlator(self, connected=False, bootstrap_sample_size=100):
        """Helper function which executes multiprocessing function to calculate the
        bootstrapped two point correlation function."""
        
        # Want to use a seed for the RNG that generates the bootstrap indices,
        # so we can generate the same indices for both terms in the correlator.
        # Also required for bootstrapping individual shifts.
        seed = np.random.randint(maxsize)

        # Compute first term, using multiprocessing
        # NB bootstrap each shift seprately to reduce peak memory requirements
        func = lambda shift: bootstrap_sample(
            self._vol_avg_two_point_correlator(shift),
            bootstrap_sample_size,
            seed=seed,
        ).mean(axis=-1)

        mp_correlator = Multiprocessing(
            func=func, generator=self.lattice.two_point_iterator
        )
        correlator_dict = mp_correlator()
        extra_dims = correlator_dict[0].shape  # bootstrap+concurrent sample dimensions

        correlator = np.array([correlator_dict[i] for i in range(self.lattice.volume)])

        # Subtract disconnected part
        if connected:  
            magnetisation_density = self.coords.mean(axis=0)
            magnetisation_density = np.abs(magnetisation_density)
            correlator -= (
                bootstrap_sample(
                    magnetisation_density,
                    bootstrap_sample_size,
                    seed=seed,  # same seed -> same bootstrap sample as first term
                )
            ).mean(
                axis=-1  # ensemble average <|m|> ...
            ) ** 2  # ...squared <|m|>^2

        return correlator.reshape(
            (self.lattice.length, self.lattice.length, *extra_dims)
        )

    @property
    def two_point_correlator(self):
        """Two point correlation function. Uses multiprocessing.
        numpy.ndarray, dimensions (*lattice_dimensions, *)"""
        return self._two_point_correlator()

    @property
    def two_point_connected_correlator(self):
        """Two point correlation function. Uses multiprocessing.
        numpy.ndarray, dimensions (*lattice_dimensions, *)"""
        return self._two_point_correlator(connected=True)

    def boot_two_point_correlator(self, connected=False, bootstrap_sample_size=100):
        """Two point correlation function for a bootstrap sample of ensembles
        numpy.ndarray, dimensions (*lattice_dimensions, *, bootstrap_sample_size)"""
        return self._boot_two_point_correlator(
            connected=connected, bootstrap_sample_size=bootstrap_sample_size
        )


class ClassicalSpinField(ScalarField):
    """
    Class for ensembles of classical spin fields.

    Parameters
    ----------
    input_coords: numpy.ndarray
        The coordinates of the spin configuration or ensemble.
        Valid inputs may have dimensions:
            - (lattice_volume, N, ensemble_size)
            - (lattice_volume, N, *, ensemble_size)
        where N is the Euclidean dimension of the spins.
        The final dimension will always be treated as the ensemble dimension in the
        context of calculating ensemble averages.
    lattice: anvil.geometry.Geometry
        Lattice object upon which the spins are defined. Must have the same number of
        sites as the 0th dimension of `input_coords`.
    """

    minimum_dimensions = 3

    def __init__(self, input_coords, lattice):
        super().__init__(input_coords, lattice)

        self.spins = self.coords  # alias

    @classmethod
    def from_spherical(cls, input_coords, lattice):
        """Instantiate class with fields in the spherical representation, i.e.
        parameterised by a set of angles. Input coordinates must have dimensions
        (lattice_size * N-1, ensemble_size)."""
        return cls(spher_to_eucl(input_coords), lattice)

    @property
    def spins(self):
        """The configuration or ensemble of N-dimensional spin vectors (alias of coords).
        numpy.ndarray, dimensions (lattice_volume, N, *, ensemble_size)"""
        return self.coords

    @unit_norm(dim=1)
    @spins.setter
    def spins(self, new):
        """Updates the spin configuration or ensemble (by updating coords), also checking
        that the spin vectors have unit norm."""
        self.coords = new  # calls coords.__set__

    @property
    def magnetisation_series(self):
        return np.sqrt((self.coords.sum(axis=0) ** 2).sum(axis=0))

    @property
    def hamiltonian(self):
        """The spin Hamiltonian for each configuration in the ensemble.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return -np.sum(
            self.spins[self.shift] * np.expand_dims(self.spins, axis=1),
            axis=2,  # sum over vector components
        ).sum(
            axis=(0, 1)
        )  # sum over dimensions and volume

    @property
    def magnetisation_sq(self):
        """The squared magnetisation for each configuration in the ensemble.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return np.sum(
            self.spins.sum(axis=0) ** 2, axis=0
        )  # sum over volume, then vector components

    def _vol_avg_two_point_correlator(self, shift):
        """Helper function which calculates the volume-averaged two point correlation
        function for a single shift, given in lexicographical form as an argument.
        """
        return np.sum(
            self.spins[shift] * self.spins,
            axis=1,  # sum over vector components
        ).mean(
            axis=0  # average over volume
        )

    def _spherical_triangle_area(self, a, b, c):
        """Helper function which calculates the surface area of a unit sphere enclosed
        by geodesics between three points on the surface. The parameters are the unit
        vectors corresponding the three points on the unit sphere.
        """
        return 2 * np.arctan2(  # arctan2 since output needs to be (-2pi, 2pi)
            np.sum(a * np.cross(b, c, axis=0), axis=0),  # numerator
            1
            + np.sum(a * b, axis=0)
            + np.sum(b * c, axis=0)
            + np.sum(c * a, axis=0),  # denominator
        )

    def _topological_charge_density(self, x0):
        """Helper function which calculates the topological charge density at a given
        lattice site for a configuration or ensemble of Heisenberg spins.
        """
        # Four points on the lattice forming a square
        x1, x3 = self.shift[x0]
        x2 = self.shift[x1, 1]
        return (
            self._spherical_triangle_area(*list(self.spins[[x0, x1, x2]]))
            + self._spherical_triangle_area(*list(self.spins[[x0, x2, x3]]))
        ) / (4 * pi)

    @property
    def topological_charge(self):
        """Topological charge of a configuration or ensemble of Heisenberg spins,
        according to the geometrical definition given in Berg and Luscher 1981.
        Uses multiprocessing.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        generator = lambda: range(self.lattice.volume)
        mp_density = Multiprocessing(
            func=self._topological_charge_density, generator=generator
        )
        charge = np.array([q for q in mp_density().values()]).sum(axis=0)
        return charge


def scalar_field(sample_training_output, training_geometry):
    return ScalarField(sample_training_output, training_geometry)


def o2_field(sample_training_output, training_geometry):
    return ClassicalSpinField.from_spherical(
        sample_training_output.reshape(training_geometry.length ** 2, 1, -1),
        training_geometry,
    )


def o3_field(sample_training_output, training_geometry):
    return ClassicalSpinField.from_spherical(
        sample_training_output.reshape(training_geometry.length ** 2, 2, -1),
        training_geometry,
    )


FIELD_OPTIONS = {
    None: scalar_field,
    "phi_four": scalar_field,
    "o2": o2_field,
    "o3": o3_field,
}

_field_ensemble = collect("field", ("training_context",))


def field_ensemble(_field_ensemble):
    return _field_ensemble[0]
