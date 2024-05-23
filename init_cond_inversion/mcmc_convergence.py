# import libraries
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import config
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils import *
save_dir = f"./exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{config.z_dim}/mcmc_convergence_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load different chains of samples into a single array
z_dim_ = config.z_dim
seed_list = [1, 2, 3, 4, 5]
prop_var_list = [0.01, 0.1, 1]
samples = np.zeros([len(seed_list)*len(prop_var_list), config.n_mcmc_samples, z_dim_])
for p_id, pp in enumerate(prop_var_list):
    for s_id, ss in enumerate(seed_list):
        data_dir = f"./exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{z_dim_}/mcmc_seed{ss}/N{config.n_mcmc_samples}_PropVar{pp}"
        z_post_samples = np.load(f"{data_dir}/z_post_samples.npy")
        print(f"z_post_samples.shape = {z_post_samples.shape} | s_id = {s_id}, seed = {ss} | p_id = {p_id}, prop_var = {pp}")
        samples[p_id*len(seed_list)+s_id, :, :] = z_post_samples

np.save(f"{save_dir}/all_post_samples.npy", samples)


# Define the function to compute Gelman-Rubin statistics
def gelman_rubin_statistic(samples):
    n_chains, n_samples, n_params = samples.shape
    # Compute the chain means and variances
    chain_means = np.mean(samples, axis=1)
    chain_vars = np.var(samples, axis=1, ddof=1)
    # Compute the mean of the chain means
    mean_of_chain_means = np.mean(chain_means, axis=0)

    # Compute the between-chain variance
    between_chain_var = np.var(chain_means, axis=0, ddof=1)
    #between_chain_var_manual = (1/(n_chains-1))*np.sum((chain_means-mean_of_chain_means)**2, axis=0)
    #print(np.allclose(between_chain_var, between_chain_var_manual))
    # Compute the within-chain variance
    within_chain_var = np.mean(chain_vars, axis=0)

    # Compute the estimated variance
    var_hat = ((n_samples-1)/n_samples)*within_chain_var + (between_chain_var/n_samples)
    # Compute the potential scale reduction factor
    psrf = np.sqrt(var_hat/within_chain_var)
    return psrf




# plot the Gelman-Rubin statistics as a function of the number of samples
n_samples_list = np.arange(0, config.n_mcmc_samples, 10000)[1:]
#n_samples_list = np.arange(0, 1000, 100)[1:]
psrf_array = np.zeros([len(n_samples_list), z_dim_])
for n_id, n_samples in enumerate(n_samples_list):
    psrf = gelman_rubin_statistic(samples[:, :n_samples, :])
    psrf_array[n_id, :] = psrf
    print(f"n_samples = {n_samples} | psrf = {psrf}")

# plot the Gelman-Rubin statistics as a function of the number of samples
plt.figure(figsize=(z_dim_*4, 4))
for z_id in range(z_dim_):
    plt.subplot(1, z_dim_, z_id+1)
    plt.plot(n_samples_list, psrf_array[:, z_id], "-o",label=f"z_{z_id}")
    if z_id == 0:
        plt.ylabel("Gelman-Rubin statistic (R_hat)")
    plt.xlabel("Number of samples")
    plt.legend()
    plt.grid(True)


# Plot the trace plot of all the chains for each dimension
plt.figure(figsize=(z_dim_*4, 4))
for z_id in range(z_dim_):
    plt.subplot(1, z_dim_, z_id+1)
    for p_id, pp in enumerate(prop_var_list):
        for s_id, ss in enumerate(seed_list):
            plt.plot(samples[p_id*len(seed_list)+s_id, :, z_id], label=f"seed={ss}, prop_var={pp}")
    plt.ylabel(f"$z_{z_id}$")
    plt.xlabel("Number of samples")
    plt.legend()
    plt.grid(True)


def compute_acf_pacf(samples, lags):
    # Calculate ACF and PACF for each dimension and chain
    n_chains, n_samples, n_dim = samples.shape
    for chain_idx in range(n_chains):
        for dim_idx in range(n_dim):
            sample_chain_dim = samples[chain_idx, :, dim_idx]
            print(f"chain_idx = {chain_idx} | dim_idx = {dim_idx}")
            if chain_idx == 1 and dim_idx == 1:
                # Calculate ACF
                acf_fig, ax_acf = plt.subplots()
                plot_acf(sample_chain_dim, ax=ax_acf, lags=lags)
                ax_acf.set_title(f"Chain {chain_idx + 1} - Dimension {dim_idx + 1} - Autocorrelation Function")
            

# Example usage:
compute_acf_pacf(samples, lags=1000)


# ======================================================================================================================
# Compute gz using all the chains
mcmc_batch_size = 5000
n_eff_samples = samples.shape[1] - int(config.burn_in*config.n_mcmc_samples)
z_post_eff_samples = samples[:, int(config.burn_in*config.n_mcmc_samples):, :].reshape(-1, z_dim_)
#assert z_post_eff_samples.shape[0] % mcmc_batch_size == 0, "zz_post_samples.shape[0] must be divisible by mcmc_batch_size"
n_batches = int(np.ceil(z_post_eff_samples.shape[0]/mcmc_batch_size))
gz = np.zeros((z_post_eff_samples.shape[0], config.x_dim, config.x_dim))
for ii in range(n_batches):
    print(f"{ii}/{n_batches}")
    zz_post_torch = torch.autograd.Variable(torch.tensor(z_post_eff_samples[ii*mcmc_batch_size:(ii+1)*mcmc_batch_size, :], dtype=torch.float32)).to(device)
    gz[ii*mcmc_batch_size:(ii+1)*mcmc_batch_size, :, :] = ((((torch.squeeze(config.generator(zz_post_torch)) + 1.) /2)*(config.max_value-config.min_value)) + config.min_value).cpu().detach().numpy()

# Compute the mean and variance of gz
x_mean = np.mean(gz, axis=0)
x_var = np.var(gz, axis=0, ddof=1)
#x_mean = np.mean(gz, axis=0)                          #(torch.mean(gz, dim=0)).detach().numpy()
#x2_3d = gz**2
#x2_mean = np.mean(x2_3d, axis=0)                      #(torch.mean(x2_3d, dim=0)).detach().numpy()
#x_var = x2_mean - (x_mean**2)
#print(np.allclose(x_var, gz_var))
#print(np.linalg.norm(x_var - gz_var))
x_std = np.sqrt(np.maximum(x_var, 0))



post_stats_plots(config.x_true, config.y_true, config.y_meas, config.x_true, x_mean, x_std, save_dir=save_dir, file_postfix="posterior", if_map=False)

# Compute the mean and variance of the samples for each dimension using all the chains
z_mean = np.mean(z_post_eff_samples, axis=0)
z_var = np.var(z_post_eff_samples, axis=0, ddof=1)
np.save(f"{save_dir}/z_mean.npy", z_mean)
np.save(f"{save_dir}/z_var.npy", z_var)
np.save(f"{save_dir}/x_mean.npy", x_mean)
np.save(f"{save_dir}/x_var.npy", x_var)
np.save(f"{save_dir}/x_std.npy", x_std)
plt.show()