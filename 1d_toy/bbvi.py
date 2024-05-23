# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 01:49:01 2022

@author: Harikrishna
"""
import os
import time
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import config
from utils import *
from models import *
from optimizers import *
plt.rcParams['font.size'] = 14
np.random.seed(1008)
t1 = time.time()
plt.close("all")


# create a save_dir inside the exps directory and name this directory based on the values of prior_mean_vec, prior_var_vec, y_meas, A, like_var
save_dir = f"./exps/dim{config.x_dim}/{config.optimizer}/postmu0_{config.true_mu_post[0,0]:.2f}_postmulast_{config.true_mu_post[-1,0]:.2f}_poststd0_{config.true_std_post[0]:.2f}_poststdlast{config.true_std_post[-1]:.2f}/bs{config.batch_size}_nit{config.n_iter}_mulr{config.mu_step_size}_glr{config.gamma_step_size}_eps{config.epsilon}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# write to file true_mu_post and the diagonal elements of cov_post
with open(f"{save_dir}/../true_post_stats.txt", "w") as f:
    f.write("true_mu_post = \n")
    f.write(str(config.true_mu_post))
    f.write("\n cov_post = \n")
    f.write(str(np.diag(config.cov_post)))
    f.write("\n true_std_post = \n")
    f.write(str(config.true_std_post))

# save config.yml to save_dir
with open(f"{save_dir}/config.yml", 'w') as config_file:
    yaml.dump(config, config_file)



print(f"batch_size = {config.batch_size} AND epsilon = {config.epsilon} =========================")  
xi_orig_1d = np.random.multivariate_normal(mean=np.zeros(config.x_dim), cov=np.eye(config.x_dim), size=1)    
batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi = data_processing(config.x_dim, config.batch_size, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)
prior = Prior(config.prior_mean_vec, config.prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec)
likelihood = Likelihood(config.y_meas, batched_y_meas, config.A, config.like_var_vec, batched_like_var_vec)
# DONE Till here... for Gamma!
if config.optimizer == "gradient_descent":
    mu_opt, gamma_opt, mu_list_np, gamma_list_np = gradient_descent(xi, xi_orig_1d, config.mu_init_guess, config.gamma_init_guess, config.n_iter, config.mu_step_size, config.gamma_step_size, prior, likelihood)  
#elif config.optimizer == "adagrad":
#    mu_opt, sigma_opt, mu_list_np, sigma_list_np = adagrad(xi, xi_orig_1d, config.mu_init_guess, config.sigma_init_guess, config.n_iter, config.mu_step_size, config.sigma_step_size, prior, likelihood)
elif config.optimizer == "adam":
    mu_opt, gamma_opt, mu_list_np, gamma_list_np = adam(xi, xi_orig_1d, config.mu_init_guess, config.gamma_init_guess, config.n_iter, config.mu_step_size, config.gamma_step_size, prior, likelihood)
#elif config.optimizer == "adam_coordinate_ascent":
#    mu_opt, sigma_opt, mu_list_np, sigma_list_np = adam_coordinate_ascent(xi, xi_orig_1d, config.mu_init_guess, config.sigma_init_guess, config.n_iter, config.mu_step_size, config.sigma_step_size, prior, likelihood)
else:
    raise ValueError("Optimizer not supported")

sigma_opt = np.exp(gamma_opt)
sigma_list_np = np.exp(gamma_list_np)
mu_l2_error, sigma_l2_error = rel_l2_error(config.true_mu_post.squeeze(), mu_opt.squeeze()), rel_l2_error(config.true_std_post.squeeze(), sigma_opt.squeeze())
mu_l1_error, sigma_l1_error = rel_l1_error(config.true_mu_post.squeeze(), mu_opt.squeeze()), rel_l1_error(config.true_std_post.squeeze(), sigma_opt.squeeze())
mu_scaled_l2_error, sigma_scaled_l2_error = rel_scaled_l2_error(config.true_mu_post.squeeze(), mu_opt.squeeze()), rel_scaled_l2_error(config.true_std_post.squeeze(), sigma_opt.squeeze())
mu_scaled_l1_error, sigma_scaled_l1_error = rel_scaled_l1_error(config.true_mu_post.squeeze(), mu_opt.squeeze()), rel_scaled_l1_error(config.true_std_post.squeeze(), sigma_opt.squeeze())

print(f"true_mu_post = {config.true_mu_post} \n mu_opt = {mu_opt} and \n\n true_std_post = {config.true_std_post} \n sigma_opt: {sigma_opt}")
print(f"=== L2 error ===")
print(f"rel. L2 error in mu = {mu_l2_error} % and \n rel. L2 error in sigma = {sigma_l2_error} %")
print(f"=== L1 error === ")
print(f"rel. L1 error in mu = {mu_l1_error} % and \n rel. L1 error in sigma = {sigma_l1_error} %")
print(f"=== Scaled L2 error ===")
print(f"rel. Scaled L2 error in mu = {mu_scaled_l2_error} % and \n rel. Scaled L2 error in sigma = {sigma_scaled_l2_error} %")
print(f"=== Scaled L1 error === ")
print(f"rel. Scaled L1 error in mu = {mu_scaled_l1_error} % and \n rel. Scaled L1 error in sigma = {sigma_scaled_l1_error} %")
print(config.optimizer)
#elbo_list = ELBO(xi, mu_list_np, sigma_list_np, n_iter, batch_size, x_dim, prior, likelihood)     
elbo_list = np.ones(config.n_iter)
var_post_samples = plots(save_dir, mu_list_np, sigma_list_np, elbo_list, config.true_mu_post.squeeze(), mu_opt.squeeze(), config.true_std_post.squeeze(), sigma_opt.squeeze(), 
                        np.mean(config.noise_var_vec), config.mu_step_size, config.gamma_step_size, config.batch_size, config.n_iter, config.x_dim, config.true_post_samples, 
                        mu_l2_error, sigma_l2_error, mu_l1_error, sigma_l1_error,  mu_scaled_l2_error, sigma_scaled_l2_error, mu_scaled_l1_error, sigma_scaled_l1_error)
#print(f"mu at the end of optimization: mu = {mu_opt} and sigma: {sigma_opt} \n \
#      Final ELBO = {elbo_list[-1]:.4f}")

print(f"total time = {time.time() - t1:.4f}")
var_post_samples = np.random.multivariate_normal(mean=mu_opt.squeeze(), cov=np.eye(config.x_dim)*sigma_opt**2, size=100000)
sample_mean = np.mean(var_post_samples, axis=0)
sample_cov = np.cov(var_post_samples.T)