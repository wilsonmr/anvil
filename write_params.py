L = 6
m_sq = -4
lam = 6.975
N_BATCH = 1000
n_affine = 16
affine_hidden_shape = (32,)
epochs = [2000,4000,6000,8000,10000,]
N_MCMC_sims = 3
target_length = 1000
tau_max = 20
i_therm = 10

training_params_dir = 'training_params/'

for value in epochs:
    with open(training_params_dir+'L6ep'+str(value)+'.txt', 'w') as outfile:
        outfile.write(f"L {L}\n")
        outfile.write(f"m_sq {m_sq}\n")
        outfile.write(f"lam {lam}\n")
        outfile.write(f"N_BATCH {N_BATCH}\n")
        outfile.write(f"n_affine {n_affine}\n")
        outfile.write(f"affine_hidden_shape {affine_hidden_shape}\n")
        outfile.write(f"epochs {value}\n")
        outfile.write(f"N_MCMC_sims {N_MCMC_sims}\n")
        outfile.write(f"target_length {target_length}\n")
        outfile.write(f"tau_max {tau_max}\n")
        outfile.write(f"i_therm {i_therm}\n")