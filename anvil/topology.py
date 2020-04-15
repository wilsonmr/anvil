
def HeisenbergField(ClassicalSpinField):
    """
    Extend ClassicalSpinField class to include topological observables defined
    exclusively for the Heisenberg, or O(3), model.
    """    
    def __init__(self, training_output, geometry):
        super().__init__(training_output, geometry, field_dimension=2)
        self.shift = self.geometry.get_shift()

    def _spherical_triangle_area(self, ilat, axis):
        """Calculates signed surface area enclosed by three points on a sphere.

        See Berg and Luscher, Nucl. Phys. B 190, 412-424, 1981.
        """

        x1 = ilat
        x2 = self.shift[axis, x1]
        x3 = self.shift[(axis + 1) % 2, x2]
        cos_angle1 = torch.sum(self.sample[:, x1, :] * self.sample[:, x2, :], dim=-1)
        cos_angle2 = torch.sum(self.sample[:, x2, :] * self.sample[:, x3, :], dim=-1)
        cos_angle3 = torch.sum(self.sample[:, x3, :] * self.sample[:, x1, :], dim=-1)
        sin_angle2 = torch.sum(
                self.field[:, x1, :]
                * torch.cross(self.sample[:, x2, :], self.sample[:, x3, :], dim=1,)
        )
        numerator = 1 + cos_angle1 + cos_angle2 + cos_angle3 + 1j * sin_angle2
        denominator = 2 * (1 + cos_angle1) * (1 + cos_angle2) * (1 + cos_angle3)
        return -2 * 1j * (torch.log(numerator) - 0.5 * torch.log(denominator))
