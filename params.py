import torch
import grid

# --------- #
#  Lattice  #
# --------- #
L = 6
D = L*L

# ---------- #
#  Training  #
# ---------- #
N_BATCH = 5000
n_affine = 8
affine_hidden_shape=(32,32)
epochs = 10000

# ---------- #
#  Sampling  #
# ---------- #
target_length = 100000
n_large = 2*target_length

# ----------------- #
#  Autocorrelation  #
# ----------------- #
tau_max = 20
i_therm = 1


# ------ #
#  phi4  #
# ------ #
class PhiFourAction(torch.nn.Module):
    """Extend the nn.Module class to return the phi^4 action given a state
    might be possible to jit compile this to make training a bit faster
    """
    def __init__(self, length, m_sq, lam):
        super(PhiFourAction, self).__init__()
        self.shift = grid.neighbours(length)
        self.lam = lam
        self.m_sq = m_sq
        self.length = length

    def forward(self, phi_state: torch.Tensor) -> torch.Tensor:
        """Given a stack of states, calculate the action for each state"""
        action = (
            (2+0.5*self.m_sq)*phi_state**2 + # phi^2 terms
            self.lam*phi_state**4 - #phi^4 term
            0.5*torch.sum(
                phi_state[:, self.shift]*phi_state.view(-1, 1, self.length**2),
                dim=1,
            ) # derivative
        ).sum(dim=1, keepdim=True) # sum across sites
        return action

m_sq = -4
lam = 6.975
action = PhiFourAction(L, m_sq, lam)

