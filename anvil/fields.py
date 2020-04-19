import torch
from math import pi

from reportengine import collect


class ScalarField:
    def __init__(self, training_output, geometry):
        self.geometry = geometry

        self.lattice_size = self.geometry.length ** 2

        self.sample = training_output
        self.sample_size = self.sample.shape[0]

    def _calc_two_point_function(self, sample):
        phi = sample.transpose(0, 1)  # sums faster over last dimension

        g_func = torch.empty((self.geometry.length, self.geometry.length))
        for i in range(self.geometry.length):
            for j in range(self.geometry.length):
                shift = self.geometry.get_shift(shifts=((i, j),), dims=((0, 1),)).view(
                    -1
                )  # make 1d

                phi_shift = phi[shift, :]

                #  Average over stack of states
                phi_mean = phi.mean(dim=1)
                phi_shift_mean = phi_shift.mean(dim=1)
                phi_shift_phi_mean = (phi_shift * phi).mean(dim=1)

                # Average over volume
                g_func[i, j] = torch.mean(
                    phi_shift_phi_mean - phi_shift_mean * phi_mean
                )
        return g_func

    def two_point_function(self, n_boot=100):
        result = []
        for n in range(n_boot):
            boot_index = torch.randint(0, self.sample_size, size=(self.sample_size,))
            resample = self.sample[boot_index, :]
            result.append(self._calc_two_point_function(resample))
        return torch.stack(result, dim=-1)

    def volume_avg_two_point_function(self):
        va_2pf = torch.empty(
            self.geometry.length, self.geometry.length, self.sample_size
        )
        mean_sq = self.sample.mean(dim=1) ** 2

        for i in range(self.geometry.length):
            for j in range(self.geometry.length):
                shift = self.geometry.get_shift(shifts=((i, j),), dims=((0, 1),)).view(
                    -1
                )  # make 1d

                va_2pf[i, j, :] = (self.sample[:, shift] * self.sample).mean(
                    dim=1
                ) - mean_sq
        return va_2pf


class ClassicalSpinField:
    def __init__(self, training_output, geometry):
        self.geometry = geometry

        self.lattice_size = self.geometry.length ** 2
        (
            self.sample_size,
            self.config_size,
        ) = training_output.shape  # not great - needed in next line
        self.spin_dimension = self.config_size // self.lattice_size + 1

        self.spins = self._spher_to_eucl(training_output)

    def _spher_to_eucl(self, training_output):
        """Take a stack of angles with shape (sample_size, (N-1) * lattice_size), where the N-1
        angles parameterise an N-spin vector on the unit (N-1)-sphere, and convert this
        to a stack of euclidean field vectors with shape (sample_size, lattice_size, N).
        """
        angles = training_output.view(self.sample_size, self.lattice_size, -1)

        spins = torch.ones(self.sample_size, self.lattice_size, self.spin_dimension)
        spins[:, :, :-1] = torch.cos(angles)
        spins[:, :, 1:] *= torch.cumprod(torch.sin(angles), dim=-1)

        return spins

    def _calc_volume_avg_two_point_function(self):
        va_2pf = torch.empty(
            (self.geometry.length, self.geometry.length, self.sample_size)
        )

        for i in range(self.geometry.length):
            for j in range(self.geometry.length):
                shift = self.geometry.get_shift(shifts=((i, j),), dims=((0, 1),)).view(
                    -1
                )
                va_2pf[i, j, :] = torch.sum(
                    self.spins[:, shift, :] * self.spins,
                    dim=2,  # sum over vector components
                ).mean(
                    dim=1
                )  # average over volume

        return va_2pf

    # Accessed by reportengine actions
    def volume_avg_two_point_function(self):
        if not hasattr(self, "_volume_avg_two_point_function"):
            self._volume_avg_two_point_function = (
                self._calc_volume_avg_two_point_function()
            )
        return self._volume_avg_two_point_function

    def two_point_function(self, n_boot=100):
        result = []
        for n in range(n_boot):
            boot_index = torch.randint(0, self.sample_size, size=(self.sample_size,))
            result.append(
                self.volume_avg_two_point_function()[:, :, boot_index].mean(dim=2)
            )
        return torch.stack(result, dim=-1)


class HeisenbergField(ClassicalSpinField):
    def __init__(self, training_output, geometry):
        super().__init__(training_output, geometry)
        self.shift = self.geometry.get_shift()

        self._topological_charge = self._calc_topological_charge()

    def _tan_half_spher_triangle(self, ilat, axis):
        x0 = ilat
        x1 = self.shift[axis, x0]
        x2 = self.shift[(axis + 1) % 2, x1]

        numerator = torch.sum(
            self.spins[:, x0, :]
            * torch.cross(self.spins[:, x1, :], self.spins[:, x2, :]),
            dim=1,
        )
        denominator = (
            1
            + torch.sum(self.spins[:, x0, :] * self.spins[:, x1, :], dim=1)
            + torch.sum(self.spins[:, x1, :] * self.spins[:, x2, :], dim=1)
            + torch.sum(self.spins[:, x2, :] * self.spins[:, x0, :], dim=1)
        )
        return numerator / denominator

    def _calc_topological_charge(self):
        charge = torch.zeros(self.sample_size)

        for ilat in range(self.lattice_size):
            charge += torch.atan(self._tan_half_spher_triangle(ilat, 0)) - torch.atan(
                self._tan_half_spher_triangle(ilat, 1)
            )
        return charge / (2 * pi)

    def topological_charge(self, n_boot=100):

        result = []
        for n in range(n_boot):
            boot_index = torch.randint(0, self.sample_size, size=(self.sample_size,))
            result.append(self._topological_charge[boot_index])
        return torch.stack(result, dim=-1)


def scalar_field(sample_training_output, training_geometry):
    return ScalarField(sample_training_output, training_geometry)


def xy_field(sample_training_output, training_geometry):
    return ClassicalSpinField(sample_training_output, training_geometry)


def heisenberg_field(sample_training_output, training_geometry):
    return HeisenbergField(sample_training_output, training_geometry)


_field_ensemble = collect("field_ensemble_action", ("training_context",))


def field_ensemble(_field_ensemble):
    return _field_ensemble[0]
