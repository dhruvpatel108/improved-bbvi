# import libraries
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import *
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.grid'] = True

bbvi_batch_size = 25
# Import ground truth posterior statistics
mcmc_post_dir = f"./exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{config.z_dim}/mcmc_convergence_results"
bbvi_post_convg_dir = f"./exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{config.z_dim}/bbvi/xi_{config.xi_sampling}/{config.optimizer}/BS{bbvi_batch_size}_bbvi_cvg"
z_post_mean = np.load(f"{mcmc_post_dir}/z_mean.npy")
z_post_var = np.load(f"{mcmc_post_dir}/z_var.npy")
z_post_std = np.sqrt(z_post_var)
x_post_mean_mcmc = np.load(f"{mcmc_post_dir}/x_mean.npy")
x_post_std_mcmc = np.load(f"{mcmc_post_dir}/x_std.npy")
x_post_mean_bbvi = np.load(f"{bbvi_post_convg_dir}/x_mean.npy")
x_post_std_bbvi = np.load(f"{bbvi_post_convg_dir}/x_std.npy")
#z_all_post_samples = np.load(f"{mcmc_post_dir}/all_post_samples.npy")


# Function to plot error in posterior statistics as a function of iteration
def error_vs_forward_solves(n_forward_solves_list, bbvi_batch_size, x_post_mean_mcmc, x_post_std_mcmc, x_post_mean_bbvi, x_post_std_bbvi, 
                            error_fn, error_fn_name, y_label):
    # make save_dir
    #bbvi_post_convg_dir = f"./exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{config.z_dim}/bbvi/xi_{config.xi_sampling}/{config.optimizer}/bbvi_convergence_results/BS{bbvi_batch_size}"
    #if not os.path.exists(bbvi_post_convg_dir):
    #    os.makedirs(bbvi_post_convg_dir)
        
    # load the MCMC error pickle file
    with open(f"{mcmc_post_dir}/error_vs_forward_solves_MC_{error_fn_name}.pickle", 'rb') as f:
        mcmc_error_dict = pickle.load(f)
    for key, value in mcmc_error_dict.items():
        print(key, value)

    error_x_mean_list = []
    error_x_std_list = []
    for n_forward in n_forward_solves_list:
        # load the BBVI files
        bbvi_post_dir = f"./exps/id{config.test_img_index}_k{config.k}_noisevar{config.noise_var}_likevar{config.like_var}/Zdim{config.z_dim}/bbvi/xi_{config.xi_sampling}/{config.optimizer}/BS{bbvi_batch_size}_N{int(n_forward/bbvi_batch_size)}_seed5"


        x_bbvi_mean = np.load(f"{bbvi_post_dir}/x_mean.npy")
        x_bbvi_std = np.load(f"{bbvi_post_dir}/x_std.npy")
        x_bbvi_map = np.load(f"{bbvi_post_dir}/x_map.npy")

        error_x_mean_list.append(error_fn(x_post_mean_bbvi, x_bbvi_mean))
        error_x_std_list.append(error_fn(x_post_std_bbvi, x_bbvi_std))

        post_stats_plots(config.x_true, config.y_true, config.y_meas, x_bbvi_map, x_bbvi_mean, x_bbvi_std, 
                         save_dir=bbvi_post_convg_dir, file_postfix=f"nf{n_forward}", if_map=True)


    # Plot error in posterior statistics as a function of the number of forward model evaluations
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].loglog(n_forward_solves_list, error_x_mean_list, 'o-', label='BBVI')
    ax[0].loglog(mcmc_error_dict['n_forward_solves_list'], mcmc_error_dict['error_x_mean_list'], 'o-', label='MCMC' )
    ax[0].set_xlabel('Number of forward solves')
    ax[0].set_ylabel('Normalized error in \n posterior mean')
    ax[0].legend()
    #ax[0].set_title("x_mean")
    ax[1].loglog(n_forward_solves_list, error_x_std_list, 'o-', label='BBVI')
    ax[1].loglog(mcmc_error_dict['n_forward_solves_list'], mcmc_error_dict['error_x_std_list'], 'o-', label='MCMC' )
    ax[1].set_xlabel('Number of forward solves')
    ax[1].set_ylabel('Normalized error in \n posterior std. dev.')
    ax[1].legend()
    #ax[1].set_title("x_std")
    plt.tight_layout()
    plt.savefig(f"{bbvi_post_convg_dir}/{error_fn_name}_error_plot.png")
    plt.show()

    # Save the error data as a dictionary
    bbvi_error_dict = {'n_forward_solves_list': n_forward_solves_list,
                    'error_x_mean_list': error_x_mean_list,
                    'error_x_std_list': error_x_std_list}
    with open(f"{bbvi_post_convg_dir}/e_vs_nf_{error_fn_name}.pickle", 'wb') as handle:
        pickle.dump(bbvi_error_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

n_forward = [150, 1500, 15000, 150000, 375000, 750000]
n_forward_solves_list = [x for x in n_forward]
print(n_forward_solves_list)
error_vs_forward_solves(n_forward_solves_list, bbvi_batch_size, x_post_mean_mcmc, x_post_std_mcmc, x_post_mean_bbvi, x_post_std_bbvi, error_fn=rel_l2_error, 
                        error_fn_name="rel_l2", y_label="$(||true - pred||_2/||true||_2)*100$")        
#error_vs_forward_solves(n_forward_solves_list, 1500, x_post_mean, x_post_std, error_fn=rel_l1_error, 
#                        error_fn_name="rel_l1", y_label="$(||true - pred||_1/||true||_1)*100$") 