# import libraries
import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config
from utils import *
from models import *
from gradient import *
plt.rcParams.update({'font.size': 18})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.close("all")

xi_orig_1d = np.random.multivariate_normal(mean=np.zeros(config.x_dim), cov=np.eye(config.x_dim), size=1)  
# =============================================================================
# optimizer
t1 = time.time()
xi_ = np.random.multivariate_normal(mean=np.zeros(config.x_dim), cov=np.eye(config.x_dim), size=config.ref_batch_size)  
prior_ = Prior(config.prior_mean_vec, config.prior_var_vec, np.repeat(config.prior_mean_vec.T, config.ref_batch_size, axis=0), 
               np.repeat(config.prior_var_vec.T, config.ref_batch_size, axis=0))
likelihood_ = Likelihood(config.y_meas, np.repeat(config.y_meas.T, config.ref_batch_size, axis=0), config.A, 
                         config.like_var_vec, np.repeat(config.like_var_vec.T, config.ref_batch_size, axis=0))




sigma_init_list = [0.70710678]
mu_init_list = [2.5]#, 5.1, 7.5, 10.]
batch_size_list = [10, 100, 1000, 10000, 100000] #[2, 10, 50, 100, 1000]

error_array = np.zeros([len(sigma_init_list), len(mu_init_list), len(batch_size_list), 2])
error_cv_array = np.zeros([len(sigma_init_list), len(mu_init_list), len(batch_size_list), 2])
error_fd_array = np.zeros([len(sigma_init_list), len(mu_init_list), len(batch_size_list), 2])
error_fd_cv_array = np.zeros([len(sigma_init_list), len(mu_init_list), len(batch_size_list), 2])

for ss, sigma_init in enumerate(sigma_init_list):

    for mm, mu_init in enumerate(mu_init_list):
        print("******************************************************************")
        mu_init_guess = np.ones([config.x_dim, 1])*mu_init
        gamma_init_guess = np.log(np.ones([config.x_dim, 1])*sigma_init)
        #gamma_init_guess = np.log(np.ones([config.x_dim, 1])*config.sigma_init_scalar)
        gradL_mu_fd, gradL_gamma_fd = fd_gradient(config.epsilon, xi_, mu_init_guess, gamma_init_guess, prior_, likelihood_, config.ref_batch_size)
        print(f"True gradient (mu): {gradL_mu_fd} \n ")

        for bb, batch_size in enumerate(batch_size_list):
            print(f"===== \n mu_init = {mu_init}, batch_size = {batch_size} =====")        
            batched_mu_t = np.repeat(mu_init_guess.T, batch_size, axis=0)
            batched_gamma_t = np.repeat(gamma_init_guess.T, batch_size, axis=0)

            batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi = \
            data_processing(config.xi_sampling, config.x_dim, batch_size, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)
            batched_xx_t = xi*np.exp(batched_gamma_t) + batched_mu_t

            prior = Prior(config.prior_mean_vec, config.prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec)
            likelihood = Likelihood(config.y_meas, batched_y_meas, config.A, config.like_var_vec, batched_like_var_vec)
            
            elbo = ELBO_batch(mu_init_guess, np.exp(gamma_init_guess), prior, likelihood, batch_size)
            print(f"ELBO = {elbo}")
            bbvi_grad_mu, bbvi_grad_gamma = bbvi_gradient(batch_size, xi, xi_orig_1d, batched_xx_t, mu_init_guess, gamma_init_guess, batched_mu_t, batched_gamma_t, prior, likelihood, 
                                                if_fd_correction=False, if_cv=False) 
            bbvi_cv_grad_mu, bbvi_cv_grad_gamma = bbvi_gradient(batch_size, xi, xi_orig_1d, batched_xx_t, mu_init_guess, gamma_init_guess, batched_mu_t, batched_gamma_t, prior, likelihood,
                                                    if_fd_correction=False, if_cv=True)
            bbvi_fd_grad_mu, bbvi_fd_grad_gamma = bbvi_gradient(batch_size, xi, xi_orig_1d, batched_xx_t, mu_init_guess, gamma_init_guess, batched_mu_t, batched_gamma_t, prior, likelihood,
                                                    if_fd_correction=True, if_cv=False)
            bbvi_fd_cv_grad_mu, bbvi_fd_cv_grad_gamma = bbvi_gradient(batch_size, xi, xi_orig_1d, batched_xx_t, mu_init_guess, gamma_init_guess, batched_mu_t, batched_gamma_t, prior, likelihood,
                                                    if_fd_correction=True, if_cv=True)
            
            
            error_array[ss, mm, bb, 0] = rel_l2_error(gradL_mu_fd, bbvi_grad_mu)
            error_array[ss, mm, bb, 1] = rel_l2_error(gradL_gamma_fd, bbvi_grad_gamma)
            error_cv_array[ss, mm, bb, 0] = rel_l2_error(gradL_mu_fd, bbvi_cv_grad_mu)
            error_cv_array[ss, mm, bb, 1] = rel_l2_error(gradL_gamma_fd, bbvi_cv_grad_gamma)
            error_fd_array[ss, mm, bb, 0] = rel_l2_error(gradL_mu_fd, bbvi_fd_grad_mu)
            error_fd_array[ss, mm, bb, 1] = rel_l2_error(gradL_gamma_fd, bbvi_fd_grad_gamma)
            error_fd_cv_array[ss, mm, bb, 0] = rel_l2_error(gradL_mu_fd, bbvi_fd_cv_grad_mu)
            error_fd_cv_array[ss, mm, bb, 1] = rel_l2_error(gradL_gamma_fd, bbvi_fd_cv_grad_gamma)
            """
            print(f"True gradient (mu): {gradL_mu_fd} \n BBVI gradient (mu): {bbvi_grad_mu} \
                \n BBVI+CV (mu): {bbvi_cv_grad_mu} \
                \n BBVI+FD (mu): {bbvi_fd_grad_mu} \
                \n BBVI+FD+CV (mu): {bbvi_fd_cv_grad_mu} \
                BBVI rel.error = {error_array[ss, mm, bb, 0]} \n BBVI + CV rel.error = {error_cv_array[ss, mm, bb, 0]} \
                BBVI + FD rel.error = {error_fd_array[ss, mm, bb, 0]} \n BBVI + FD + CV rel.error = {error_fd_cv_array[ss, mm, bb, 0]}")
            print(f"True gradient (gamma): {gradL_gamma_fd} \n BBVI gradient (gamma): {bbvi_grad_gamma} \
                    \n BBVI+CV (gamma): {bbvi_cv_grad_gamma} \
                    \n BBVI+FD (gamma): {bbvi_fd_grad_gamma} \
                    \n BBVI+FD+CV (gamma): {bbvi_fd_cv_grad_gamma} \
                    BBVI rel.error = {error_array[ss, mm, bb, 1]} \n BBVI + CV rel.error = {error_cv_array[ss, mm, bb, 1]} \
                    BBVI + FD rel.error = {error_fd_array[ss, mm, bb, 1]} \n BBVI + FD + CV rel.error = {error_fd_cv_array[ss, mm, bb, 1]}")
            """

    save_dir = f"./exps/xdim{config.x_dim}_xi{config.xi_sampling}/sigma{sigma_init}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if config.np_save:
        np.save(f"{save_dir}/error_bbvi.npy", error_array)
        np.save(f"{save_dir}/error_bbvi_cv.npy", error_cv_array)
        np.save(f"{save_dir}/error_bbvi_fd.npy", error_fd_array)
        np.save(f"{save_dir}/error_bbvi_fd_cv.npy", error_fd_cv_array)

#np.save(f"./exps/error_bbvi_xdim{config.x_dim}_xi{config.xi_sampling}.npy", error_array)
#np.save(f"./exps/error_bbvi_cv_xdim{config.x_dim}_xi{config.xi_sampling}.npy", error_cv_array)
#np.save(f"./exps/error_bbvi_fd_xdim{config.x_dim}_xi{config.xi_sampling}.npy", error_fd_array)
#np.save(f"./exps/error_bbvi_fd_cv_xdim{config.x_dim}_xi{config.xi_sampling}.npy", error_fd_cv_array)




for ss, sigma_init in enumerate(sigma_init_list):
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    for mm, mu_init in enumerate(mu_init_list):
        #if mu_init != 5:    
        color = next(ax._get_lines.prop_cycler)['color']
        plt.semilogy(error_array[ss, mm, :, 0], "o-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init}")
        plt.semilogy(error_cv_array[ss, mm, :, 0], "x-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init} (CV)")
        plt.semilogy(error_fd_array[ss, mm, :, 0], "s-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init} (FD)")
        plt.semilogy(error_fd_cv_array[ss, mm, :, 0], "d-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init} (FD+CV)")
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(np.arange(len(batch_size_list)), batch_size_list)  
    plt.ylabel(r"$\dfrac{|(\nabla_{\mu} L)_{BBVI} - (\nabla_{\mu} L)_{FD}|}{|(\nabla_{\mu} L)_{FD}|}*100$")
    plt.xlabel("Batch size")
    plt.title(rf"$\sigma$ = {sigma_init}, $x_{{dim}}$ = {config.x_dim}, $\xi$-sampling = {config.xi_sampling}")
    plt.tight_layout()

    # sigma
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    for mm, mu_init in enumerate(mu_init_list):
        #if mu_init != 5:    
        color = next(ax._get_lines.prop_cycler)['color']
        plt.semilogy(error_array[ss, mm, :, 1], "o-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init}")
        plt.semilogy(error_cv_array[ss, mm, :, 1], "x-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init} (CV)")
        plt.semilogy(error_fd_array[ss, mm, :, 1], "s-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init} (FD)")
        plt.semilogy(error_fd_cv_array[ss, mm, :, 1], "d-", color=color, markerfacecolor='none', linewidth=2, markersize=12, label=f"$\mu$ = {mu_init} (FD+CV)")
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(np.arange(len(batch_size_list)), batch_size_list)
    plt.ylabel(r"$\dfrac{|(\nabla_{\gamma} L)_{BBVI} - (\nabla_{\gamma} L)_{FD}|}{|(\nabla_{\gamma} L)_{FD}|}*100$")
    plt.xlabel("Batch size")
    plt.title(rf"$\sigma$ = {sigma_init}, $x_{{dim}}$ = {config.x_dim}, $\xi$-sampling = {config.xi_sampling}")
    plt.tight_layout()

print(f"total time = {time.time() - t1:.4f}")
plt.show()




