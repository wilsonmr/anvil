import torch
import grid

# --------- #
#  Lattice  #
# --------- #
L = 4
D = L*L

# ---------- #
#  Training  #
# ---------- #
N_BATCH = 1000
n_affine = 12
affine_hidden_shape=(32,)
epochs = 100

# Directory for saving models. Include '/'
model_dir = 'models/'

# Run a few MCMC simulations every few epochs as another convergence check
epochs_sample = epochs//10
N_MCMC_sims = 3

# Directory for saving convergence data. Include '/'
training_data_dir = 'training_data/'

# ---------- #
#  Sampling  #
# ---------- #
target_length = 1000
n_large = 5*target_length

# ----------------- #
#  Autocorrelation  #
# ----------------- #
tau_max = 20 # longest computed autocorrelation 'time'
i_therm = 10 # only start measuring autocorrelation after i_therm configurations


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

