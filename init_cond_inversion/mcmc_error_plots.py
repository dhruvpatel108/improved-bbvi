# import libraries
import numpy as np
import matplotlib.pyplot as plt
import config
import torch
import pickle
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.grid'] = True

# Import ground truth posterior statistics
mcmc_post_dir = f"./exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{config.z_dim}/mcmc_convergence_results"
z_post_mean = np.load(f"{mcmc_post_dir}/z_mean.npy")
z_post_var = np.load(f"{mcmc_post_dir}/z_var.npy")
z_post_std = np.sqrt(z_post_var)
x_post_mean = np.load(f"{mcmc_post_dir}/x_mean.npy")
x_post_std = np.load(f"{mcmc_post_dir}/x_std.npy")
z_all_post_samples = np.load(f"{mcmc_post_dir}/all_post_samples.npy")


def z_to_post_stats(z, z_post_mean, z_post_std, x_post_mean, x_post_std, n_forward_solves):
    # z-stats
    z_post_mean_ = np.mean(z, axis=0)
    z_post_std_ = np.std(z, axis=0, ddof=1)
    # x-stats
    z_torch = torch.from_numpy(z).float().to(device)
    gz = (((config.generator(z_torch)+1.)/2)*(config.max_value-config.min_value)+config.min_value).detach().cpu().numpy().squeeze()
    print(f"n_forward_solves: {n_forward_solves}, gz.shape: {gz.shape}")
    x_post_mean_ = np.mean(gz, axis=0)
    x_post_std_ = np.std(gz, axis=0, ddof=1)
    return z_post_mean_, z_post_std_, x_post_mean_, x_post_std_
    


# Function to compute the error in posterior statistics as a function of the number of forward model evaluations
def error_vs_forward_solves(n_forward_solves_list, z_post_mean, z_post_std, x_post_mean, x_post_std, z_all_post_samples, 
                            if_single_chain, error_fn, error_fn_name, y_label):
    if if_single_chain:
        z_post_samples = z_all_post_samples[0, int(config.burn_in*config.n_mcmc_samples):, :]
    else:
        z_post_samples = z_all_post_samples[:, int(config.burn_in*config.n_mcmc_samples):, :]
        n_chains = z_post_samples.shape[0]

    error_z_mean_list = []
    error_z_std_list = []
    error_x_mean_list = []
    error_x_std_list = []    
    mcmc_batch_size = 20000
    for n_forward_solves in n_forward_solves_list:
        if if_single_chain:
            z_post_samples_subset = z_post_samples[:n_forward_solves, :]
        else:
            z_post_samples_subset = z_post_samples[:, :int(n_forward_solves/n_chains), :].reshape(-1, config.z_dim)

        print(f"z_post_samples_subset.shape: {z_post_samples_subset.shape}, {int(n_forward_solves/n_chains)}")
        if n_forward_solves > mcmc_batch_size:
            n_batches = int(np.ceil(n_forward_solves/mcmc_batch_size))
            z_post_mean_array = np.zeros((n_batches, config.z_dim))
            z_post_std_array = np.zeros((n_batches, config.z_dim))
            x_post_mean_array = np.zeros((n_batches, config.x_dim, config.x_dim))
            x_post_std_array = np.zeros((n_batches, config.x_dim, config.x_dim))
            for ii in range(n_batches):
                if ii == n_batches-1:
                    z_post_samples_subset_b = z_post_samples_subset[ii*mcmc_batch_size:]
                else:
                    z_post_samples_subset_b = z_post_samples_subset[ii*mcmc_batch_size:(ii+1)*mcmc_batch_size]
                print(f"ii/n_batches: {ii}/{n_batches}")
                z_post_mean_b, z_post_std_b, x_post_mean_b, x_post_std_b = z_to_post_stats(z_post_samples_subset_b, z_post_mean, z_post_std, x_post_mean, x_post_std, n_forward_solves)
                z_post_mean_array[ii, :] = z_post_mean_b
                z_post_std_array[ii, :] = z_post_std_b
                x_post_mean_array[ii, :, :] = x_post_mean_b
                x_post_std_array[ii, :, :] = x_post_std_b
            z_post_mean_ = np.mean(z_post_mean_array, axis=0)
            z_post_std_ = np.mean(z_post_std_array, axis=0)
            x_post_mean_ = np.mean(x_post_mean_array, axis=0)
            x_post_std_ = np.mean(x_post_std_array, axis=0)

        else:   
            z_post_mean_, z_post_std_, x_post_mean_, x_post_std_ = z_to_post_stats(z_post_samples_subset, z_post_mean, z_post_std, x_post_mean, x_post_std, n_forward_solves)
      

        error_z_mean_list.append(error_fn(z_post_mean, z_post_mean_))
        error_z_std_list.append(error_fn(z_post_std, z_post_std_))
        error_x_mean_list.append(error_fn(x_post_mean, x_post_mean_))
        error_x_std_list.append(error_fn(x_post_std, x_post_std_))

        post_stats_plots(config.x_true, config.y_true, config.y_meas, config.x_true, x_post_mean_, x_post_std_, 
                         save_dir=mcmc_post_dir, file_postfix=f"MC_nforward{n_forward_solves}", if_map=False)



    # Plot error in posterior statistics as a function of the number of forward model evaluations
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].loglog(n_forward_solves_list, error_z_mean_list, 'o-', label='z mean')
    ax[0, 0].set_xlabel('Number of forward solves')
    ax[0, 0].set_ylabel(f'{y_label}')
    ax[0, 0].legend()
    ax[0, 1].loglog(n_forward_solves_list, error_z_std_list, 'o-', label='z std')
    ax[0, 1].set_xlabel('Number of forward solves')
    ax[0, 1].set_ylabel(f'{y_label}')
    ax[0, 1].legend()
    ax[1, 0].loglog(n_forward_solves_list, error_x_mean_list, 'o-', label='x mean')
    ax[1, 0].set_xlabel('Number of forward solves')
    ax[1, 0].set_ylabel(f'{y_label}')
    ax[1, 0].legend()
    ax[1, 1].loglog(n_forward_solves_list, error_x_std_list, 'o-', label='x std')
    ax[1, 1].set_xlabel('Number of forward solves')
    ax[1, 1].set_ylabel(f'{y_label}')
    ax[1, 1].legend()
    plt.tight_layout()
    #plt.grid(True)
    plt.savefig(f"{mcmc_post_dir}/error_vs_forward_solves_MC_{error_fn_name}.png")
    plt.show()

    # Save the error data as a dictionary
    error_dict = {'n_forward_solves_list': n_forward_solves_list,
                    'error_z_mean_list': error_z_mean_list,
                    'error_z_std_list': error_z_std_list,
                    'error_x_mean_list': error_x_mean_list,
                    'error_x_std_list': error_x_std_list}
    with open(f"{mcmc_post_dir}/error_vs_forward_solves_MC_{error_fn_name}.pickle", 'wb') as handle:
        pickle.dump(error_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


n_forward = [10, 100, 1000, 10000, 25000, 50000]
n_forward = [x*15 for x in n_forward]
print(n_forward)
error_vs_forward_solves(n_forward, z_post_mean, z_post_std, x_post_mean, x_post_std, z_all_post_samples,
                        if_single_chain=False, error_fn=rel_l2_error, error_fn_name="rel_l2", y_label="$(||true - pred||_2/||true||_2)*100$")
#error_vs_forward_solves(n_forward, z_post_mean, z_post_std, x_post_mean, x_post_std, z_all_post_samples,
#                        if_single_chain=False, error_fn=rel_l1_error, error_fn_name="rel_l1", y_label="$(||true - pred||_1/||true||_1)*100$")
#error_vs_forward_solves(n_forward, z_post_mean, z_post_std, x_post_mean, x_post_std, z_all_post_samples,
#                        if_single_chain=False, error_fn=rel_scaled_l2_error, error_fn_name="rel_scaled_l2", y_label="$(||true - pred/true||_2)*100$")
#error_vs_forward_solves(n_forward, z_post_mean, z_post_std, x_post_mean, x_post_std, z_all_post_samples,
#                        if_single_chain=False, error_fn=rel_scaled_l1_error, error_fn_name="rel_scaled_l1", y_label="$(||true - pred/true||_1)*100$")
