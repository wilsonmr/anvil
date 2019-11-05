import torch
import grid
import sys

# Directory for saving models. Include '/'
model_dir = 'models/'

# Directory containing input training parameters. Include '/'
training_param_dir = 'training_params/'
params_file = sys.argv[1]

params = open(training_param_dir+params_file+'.txt').readlines()
params = [param.strip('\n').split(' ') for param in params]

# Directory for saving convergence data. Include '/'
training_data_dir = 'training_data/'

# --------- #
#  Lattice  #
# --------- #
L = int(params[0][1])
D = L*L

# ---------- #
#  Training  #
# ---------- #
N_BATCH = int(params[3][1])
n_affine = int(params[4][1])
affine_hidden_shape = eval(params[5][1])
epochs = int(params[6][1])

# Run a few MCMC simulations every few epochs as another convergence check
epochs_sample = epochs//10
N_MCMC_sims = 1 #int(params[7][1])

# ---------- #
#  Sampling  #
# ---------- #
target_length = int(params[8][1])
n_large = 5*target_length

# ----------------- #
#  Autocorrelation  #
# ----------------- #
tau_max = int(params[8][1]) # longest computed autocorrelation 'time'
i_therm = int(params[9][1]) # only start measuring autocorrelation after i_therm configurations


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

m_sq = int(params[1][1])
lam = float(params[2][1])
action = PhiFourAction(L, m_sq, lam)

