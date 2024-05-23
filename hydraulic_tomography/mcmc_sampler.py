# Import libraries
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import *
from models import *
from mcmc_stats import *
np.random.seed(config.seed_no)
# ==============================================================================
xi_orig_1d = np.random.multivariate_normal(mean=np.zeros(config.z_dim), cov=np.eye(config.z_dim), size=1)    
batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi = data_processing(
    config.xi_sampling, config.z_dim, config.batch_size, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)


prior = Prior(config.prior_mean_vec, config.prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec)
likelihood = Likelihood(config.y_meas, batched_y_meas, config.like_var_vec, batched_like_var_vec, config.batch_size)

def log_post(z):
    log_prior_ = prior.log_prior_sample(z)
    log_like_ = likelihood.log_likelihood_sample(z)
    return log_prior_ + log_like_



# Define the function to sample from the posterior using MCMC
def mcmc_samples(n, prop_var):
    z_cur = np.random.multivariate_normal(mean=np.zeros((config.z_dim)), cov=np.eye(config.z_dim), size=1).reshape(config.z_dim, 1)
    #np.zeros([config.z_dim, 1])
    post_cur = log_post(z_cur)    
    innov = np.random.multivariate_normal(mean=np.zeros((config.z_dim)), cov=np.eye(config.z_dim)*prop_var, size=n)
    #print(np.expand_dims(innov[0, :], axis=1).shape)
    u = np.random.uniform(size=n)
    z_trace = []
    for t in range(n):
        print(t)
        z_prop = z_cur + np.expand_dims(innov[t, :], axis=1)
        post_prop = log_post(z_prop)
        alpha = np.exp(post_prop - post_cur)
        if u[t] <= alpha:
            z_cur = z_prop
            post_cur = post_prop
        z_trace.append(z_cur)

    return np.array(z_trace).squeeze()



# ==============================================================================
t1 = time.time()
z_post_samples = mcmc_samples(config.n_mcmc_samples, config.mcmc_prop_var)
t2 = time.time()
mcmc_post_processing(z_post_samples, t1, t2, f"mcmc_seed{config.seed_no}")




"""
zz_post_samples = z_post_samples[int(config.burn_in*z_post_samples.shape[0]):, :]
zz_post_torch = torch.autograd.Variable(torch.tensor(zz_post_samples, dtype=torch.float32))
gz = (((torch.squeeze(config.generator(zz_post_torch)) + 1.) /2)*(config.max_value-config.min_value)) + config.min_value
print(gz.shape)
z_post_mean = np.mean(zz_post_samples, axis=0)
z_post_std = np.std(zz_post_samples, axis=0)
print(f"z_post_mean = {z_post_mean}, z_post_std = {z_post_std}")
save_dir = f"{config.data_dir}/mcmc/N{config.n_mcmc_samples}_PropVar{config.mcmc_prop_var}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure(figsize=(15, 4))
for ii in range(10):
    plt.subplot(2, 5, ii+1)
    plt.imshow(gz[ii, :, :].detach().numpy(), cmap="viridis")
plt.savefig(f"{save_dir}/gen_samples.png")

x2_3d = gz**2
x_mean = (torch.mean(gz, dim=0)).detach().numpy()
x2_mean = (torch.mean(x2_3d, dim=0)).detach().numpy()
x_var = x2_mean - (x_mean**2)
x_std = np.sqrt(np.maximum(x_var, 0))
post_stats_plots(config.x_true, config.y_true, config.y_meas, config.x_true, x_mean, x_std, save_dir=save_dir)

plt.figure(figsize=(15, 15))
for ii in range(5):
    plt.subplot(1, 5, ii+1)
    plt.plot(zz_post_samples[:, ii])
plt.savefig(f"{save_dir}/z_trace.png")
    
plt.figure(figsize=(15, 15))
for ii in range(5):
    plt.subplot(1, 5, ii+1)
    plt.hist(zz_post_samples[:, ii])
plt.savefig(f"{save_dir}/z_hist.png")

np.save(f"{save_dir}/zeff_post_samples.npy", zz_post_samples)
np.save(f"{save_dir}/x_mean.npy", x_mean)
np.save(f"{save_dir}/x_std.npy", x_std)
plt.show()
"""