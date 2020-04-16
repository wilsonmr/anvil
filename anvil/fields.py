import torch
from math import pi


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
    def __init__(self, training_output, geometry, field_dimension):
        self.geometry = geometry
        self.field_dimension = field_dimension

        self.lattice_size = (
            self.geometry.length ** 2
        )  # need to do this in a dim indep way
        self.shift = self.geometry.get_shift()

        self.sample_size = training_output.shape[0]  # not great - needed in next line
        self.sample = self._spher_to_eucl(training_output)
        self._modulus_check()
        self._volume_avg_two_point_function = self._calc_volume_avg_two_point_function()

        # Topological charge density formula valid for O(3) only
        if field_dimension == 2:
            self._topological_charge = self._calc_topological_charge()

    def _spher_to_eucl(self, training_output):
        """Take a stack of angles with shape (sample_size, (N-1) * lattice_size), where the N-1
        angles parameterise an N-spin vector on the unit (N-1)-sphere, and convert this
        to a stack of euclidean field vectors with shape (sample_size, lattice_size, N).
        """
        angles = training_output.view(
            self.sample_size, self.lattice_size, self.field_dimension
        )

        vectors = torch.ones(
            self.sample_size, self.lattice_size, self.field_dimension + 1
        )
        vectors[:, :, :-1] = torch.cos(angles)
        vectors[:, :, 1:] *= torch.cumprod(torch.sin(angles), dim=-1)

        return vectors

    def _modulus_check(self):
        modulus = torch.sum(self.sample ** 2, dim=-1)
        if torch.any(torch.abs(modulus - 1) > 0.01):
            print("smeg")  # TODO

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
                    self.sample[:, shift, :] * self.sample,
                    dim=2,  # sum over vector components
                ).mean(
                    dim=1
                )  # average over volume

        return va_2pf

    def _tan_half_spher_triangle(self, ilat, axis):
        x0 = ilat
        x1 = self.shift[axis, x0]
        x2 = self.shift[(axis + 1) % 2, x1]

        numerator = torch.sum(
            self.sample[:, x0, :]
            * torch.cross(self.sample[:, x1, :], self.sample[:, x2, :]),
            dim=1,
        )
        denominator = (
            1
            + torch.sum(self.sample[:, x0, :] * self.sample[:, x1, :], dim=1)
            + torch.sum(self.sample[:, x1, :] * self.sample[:, x2, :], dim=1)
            + torch.sum(self.sample[:, x2, :] * self.sample[:, x0, :], dim=1)
        )
        return numerator / denominator

    def _calc_topological_charge(self):
        charge = torch.zeros(self.sample_size)

        for ilat in range(self.lattice_size):
            charge += torch.atan(self._tan_half_spher_triangle(ilat, 0)) + torch.atan(
                self._tan_half_spher_triangle(ilat, 1)
            )
        return charge / (2 * pi)

    # Accessed by reportengine actions
    def volume_avg_two_point_function(self):
        return self._volume_avg_two_point_function

    def two_point_function(self, n_boot=100):

        result = []
        for n in range(n_boot):
            boot_index = torch.randint(0, self.sample_size, size=(self.sample_size,))
            result.append(
                self._volume_avg_two_point_function[:, :, boot_index].mean(dim=2)
            )
        return torch.stack(result, dim=-1)

    def topological_charge(self, n_boot=100):

        result = []
        for n in range(n_boot):
            boot_index = torch.randint(0, self.sample_size, size=(self.sample_size,))
            result.append(self._topological_charge[boot_index])
        return torch.stack(result, dim=-1)
