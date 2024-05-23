# import libraries
import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models_sagan import Generator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config
from utils import *
from models import *
from optimizers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(f"batch_size = {config.batch_size} AND epsilon = {config.epsilon} =========================")  
# Prepare data
xi_orig_1d = np.random.multivariate_normal(mean=np.zeros(config.z_dim), cov=np.eye(config.z_dim), size=1)    
batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi = data_processing(
    config.xi_sampling, config.z_dim, config.batch_size, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)
elbo_batched_prior_mean_vec, elbo_batched_prior_var_vec, elbo_batched_like_var_vec, elbo_batched_y_meas, _ = data_processing(
    config.xi_sampling, config.z_dim, config.elbo_batch_size, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)
#print(f"batched_y_meas.shape = {batched_y_meas.shape}")

# Define prior and likelihood
prior = Prior(config.prior_mean_vec, config.prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec)
likelihood = Likelihood(config.y_meas, batched_y_meas, config.like_var_vec, batched_like_var_vec, config.batch_size)
prior_elbo = Prior(config.prior_mean_vec, config.prior_var_vec, elbo_batched_prior_mean_vec, elbo_batched_prior_var_vec)
likelihood_elbo = Likelihood(config.y_meas, elbo_batched_y_meas, config.like_var_vec, elbo_batched_like_var_vec, config.elbo_batch_size)

# =============================================================================
# optimizer
if config.optimizer == "gradient_descent":
    mu_opt, gamma_opt, mu_list_np, gamma_list_np, elbo_list = gradient_descent(xi, xi_orig_1d, config.mu_init_guess, config.gamma_init_guess, 
                                                                    config.n_iter, config.mu_step_size, config.gamma_step_size, prior, likelihood, 
                                                                    prior_elbo, likelihood_elbo, if_fd_correction=False, if_cv=True)  
elif config.optimizer == "adam":
    mu_opt, gamma_opt, mu_list_np, gamma_list_np, elbo_list = adam(xi, xi_orig_1d, config.mu_init_guess, config.gamma_init_guess, 
                                                        config.n_iter, config.mu_step_size, config.gamma_step_size, prior, likelihood, prior_elbo, likelihood_elbo,
                                                        if_fd_correction=False, if_cv=True)
else:
    raise ValueError("Optimizer not supported")

sigma_opt = np.exp(gamma_opt)
sigma_list_np = np.exp(gamma_list_np)
print(sigma_list_np.shape)
print(f"mu_opt = {mu_opt}, \n sigma_opt = {sigma_opt}")
cov_opt = np.diag(sigma_opt.squeeze()**2)
print(f"cov_opt = {cov_opt}")

# =============================================================================
# generate sample from the variational posterior
zz_opt = np.random.multivariate_normal(mean=mu_opt.squeeze(), cov=cov_opt, size=config.n_samples)
zz_opt = torch.from_numpy(zz_opt).float()
zz_opt = zz_opt.view(config.n_samples, -1).to(device)
stats_batch_size = 10000
n_batches = int(np.ceil(config.n_samples/stats_batch_size))
gz_opt_np = np.zeros((config.n_samples, config.x_dim, config.x_dim))
for ii in range(n_batches):
    if ii == n_batches-1:
        zz_opt_batch = zz_opt[ii*stats_batch_size:, :]
        gz_opt_batch = ((config.generator(zz_opt_batch)+1)/2)*(config.max_value-config.min_value) + config.min_value
        gz_opt_np[ii*stats_batch_size:, :, :] = gz_opt_batch.cpu().detach().numpy().squeeze()
    else:
        zz_opt_batch = zz_opt[ii*stats_batch_size:(ii+1)*stats_batch_size, :]
        gz_opt_batch = ((config.generator(zz_opt_batch)+1)/2)*(config.max_value-config.min_value) + config.min_value
        gz_opt_np[ii*stats_batch_size:(ii+1)*stats_batch_size, :, :] = gz_opt_batch.cpu().detach().numpy().squeeze()
  

#gz_opt = ((config.generator(zz_opt)+1)/2)*(config.max_value-config.min_value) + config.min_value
#gz_opt = gz_opt_np.view(config.n_samples, config.x_dim, config.x_dim)
#gz_opt = gz_opt.cpu().detach().numpy()
        
# =============================================================================
# compute mean and standard deviation of the generated sample
x_mean = np.mean(gz_opt_np, axis=0)
x_var = np.var(gz_opt_np, axis=0, ddof=1)
x_std = np.sqrt(x_var)
x_map = (((config.generator(torch.from_numpy(mu_opt).float().view(1, -1).to(device))+1)/2)*
         (config.max_value-config.min_value) + config.min_value).cpu().detach().numpy().squeeze()

save_dir = f"exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{config.z_dim}/bbvi/xi_{config.xi_sampling}/{config.optimizer}/BS{config.batch_size}_N{config.n_iter}_seed{config.seed_no}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
post_stats_plots(config.x_true, config.y_true, config.y_meas, x_map, x_mean, x_std, save_dir=save_dir, file_postfix=f"_nf{config.batch_size*config.n_iter}", if_map=True)
variational_params_plots(elbo_list, mu_list_np, sigma_list_np, save_dir=save_dir)
plt.figure(figsize=(15, 4))
for ii in range(10):
    plt.subplot(2, 5, ii+1)
    plt.imshow(gz_opt_np[ii, :, :], cmap="viridis")
plt.savefig(f"{save_dir}/gen_samples.png")

np.save(f"{save_dir}/elbo_list.npy", np.array(elbo_list))
np.save(f"{save_dir}/x_map.npy", x_map)
np.save(f"{save_dir}/x_mean.npy", x_mean)
np.save(f"{save_dir}/x_std.npy", x_std)
np.save(f"{save_dir}/mu_list.npy", mu_list_np)
np.save(f"{save_dir}/sigma_list.npy", sigma_list_np)
# create a text file to store mu_opt and sigma_opt and corresponding elbo
with open(f"{save_dir}/final_elbo.txt", "w") as f:
    f.write(f"{elbo_list[-1]}")

# create a text file to store all the parameters of params.yaml file
with open(f"{save_dir}/params.txt", "w") as f:
    for key, value in config.config.items():
        f.write(f"{key} = {value}\n")

#plt.show()
# =============================================================================
