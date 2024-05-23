# Import libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import *


# ==============================================================================
def mcmc_post_processing(z_post_samples, t1, t2, save_dir_prefix):
    save_dir = f"exps/Zdim{config.z_dim}_likevar{config.like_var}/{save_dir_prefix}/N{config.n_mcmc_samples}_PropVar{config.mcmc_prop_var}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(f"{save_dir}/z_post_samples.npy", z_post_samples)
    







    zz_post_samples = z_post_samples[int(config.burn_in*z_post_samples.shape[0]):, :]
    print(f"zz_post_samples.shape = {zz_post_samples.shape}")
    
    # trace plots and histograms
    plt.figure(figsize=(15, 15))
    for ii in range(config.z_dim):
        plt.subplot(1, config.z_dim, ii+1)
        plt.plot(zz_post_samples[:, ii])
    plt.savefig(f"{save_dir}/z_trace.png")
        
    plt.figure(figsize=(15, 15))
    for ii in range(config.z_dim):
        plt.subplot(1, config.z_dim, ii+1)
        plt.hist(zz_post_samples[:, ii])
    plt.savefig(f"{save_dir}/z_hist.png")

    # store gz in batch-wise manner
    mcmc_batch_size = zz_post_samples.shape[0]
    assert zz_post_samples.shape[0] % mcmc_batch_size == 0, "zz_post_samples.shape[0] must be divisible by mcmc_batch_size"
    n_batches = int(np.ceil(zz_post_samples.shape[0]/mcmc_batch_size))
    gz = np.zeros((zz_post_samples.shape[0], config.box_z_dim, config.box_x_dim))
    for ii in range(n_batches):
        zz_post_torch = torch.autograd.Variable(torch.tensor(zz_post_samples[ii*mcmc_batch_size:(ii+1)*mcmc_batch_size, :], dtype=torch.float32)).to(device)
        gz[ii*mcmc_batch_size:(ii+1)*mcmc_batch_size, :, :] = ((((torch.squeeze(config.generator(zz_post_torch)) + 1.) /2)*(config.max_value-config.min_value)) + config.min_value).detach().cpu().numpy()

    #zz_post_torch = torch.autograd.Variable(torch.tensor(zz_post_samples, dtype=torch.float32))
    #gz = (((torch.squeeze(config.generator(zz_post_torch)) + 1.) /2)*(config.max_value-config.min_value)) + config.min_value
    print(f"gz.shape = {gz.shape}")
    #z_post_mean = np.mean(zz_post_samples, axis=0)
    #z_post_std = np.std(zz_post_samples, axis=0)
    #print(f"z_post_mean = {z_post_mean}, z_post_std = {z_post_std}")
    

    plt.figure(figsize=(15, 4))
    gz_plt_ind = np.random.choice(gz.shape[0], 10, replace=False)
    for ii in range(10):
        plt.subplot(2, 5, ii+1)
        plt.imshow(gz[gz_plt_ind[ii], :, :], cmap="viridis")
    plt.savefig(f"{save_dir}/gen_samples.png")

    x2_3d = gz**2
    x_mean = np.mean(gz, axis=0)                          #(torch.mean(gz, dim=0)).detach().numpy()
    x2_mean = np.mean(x2_3d, axis=0)                      #(torch.mean(x2_3d, dim=0)).detach().numpy()
    x_var = x2_mean - (x_mean**2)
    x_std = np.sqrt(np.maximum(x_var, 0))
    post_stats_plots(config.x_true, config.x_true, x_mean, x_std, save_dir=save_dir, file_postfix=f"_nf{config.batch_size*config.n_iter}", if_map=False)
    np.save(f"{save_dir}/x_mean.npy", x_mean)
    np.save(f"{save_dir}/x_std.npy", x_std)
    plt.show()    
    # =============================================================================
    print(f"Time taken = {t2-t1} seconds")
    mean_error = rel_l2_error(config.x_true, x_mean)
    # percentage of pixels where true is within 95% confidence interval of the estimated mean
    coverage = np.mean((config.x_true >= x_mean - 1.96*x_std) & (config.x_true <= x_mean + 1.96*x_std))
    print(f"Mean error = {mean_error}, Coverage = {coverage}")