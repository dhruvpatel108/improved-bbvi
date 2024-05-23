# Import libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import *
from models import *
from mcmc_post_processing import *

# ==============================================================================
xi_orig_1d = np.random.multivariate_normal(mean=np.zeros(config.z_dim), cov=np.eye(config.z_dim), size=1)    
batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi = data_processing(
    config.z_dim, 1, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)
prior = Prior(config.prior_mean_vec, config.prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec)
likelihood = Likelihood(config.y_meas, batched_y_meas, config.like_var_vec, batched_like_var_vec)
# ==============================================================================

def U_and_gradientU(z):#, generator, Ft, y_meas_torch, low_temp, high_temp, like_var):
    """
    Compute potential energy and its gradient for given state vector z using forward model
    Input
    z : position vector of shape [z_dim, 1]
    Output
    U := -log prob(z) of type scalar     
    grad_U := gradient of U of shape [z_dim, 1]
    """
    z0 = torch.autograd.Variable(torch.tensor(z, dtype=torch.float32), requires_grad=True)
    gz = (((torch.squeeze(config.generator(z0.T)) + 1.) /2)*(config.high_temp-config.low_temp)) + config.low_temp
    gz_int = gz[1:-1, 1:-1].reshape([-1, 1])

    # potential energy calculation
    y_pred = config.Ft @ gz_int
    diff = y_pred - config.y_meas_torch
    log_like = -((diff.T @ diff)/(2*config.like_var))
    log_prior = -((z0.T @ z0)/2)
    U = (-log_like - log_prior).squeeze()
    # gradient calculation
    gradU = torch.autograd.grad(U, z0)

    return U.detach().numpy(), gradU[0].detach().numpy()



def hmc(epsilon, L, z_0, d):
    """
    z: position variable
    p: momentum variable
    """    
    z = z_0    
    p_0 = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=1).T
    p = p_0

    # make half step for momentum at the beginning
    U0, grad_U = U_and_gradientU(z)
    p = p - (epsilon/2)*grad_U
    
    # alternate full steps for position and momentum
    for ii in range(L):       
        z = z + epsilon*p        
        # update full step for p except for last step
        if ii != L-1:
            _, grad_U = U_and_gradientU(z)
            p = p - epsilon*grad_U
        
    # updte last half step for p
    UL, grad_U = U_and_gradientU(z)
    p = p - (epsilon/2)*grad_U

    # negate momentum to make it symmetric
    p = -p
    
    # evaluate kinetric energy(K), potential energy(U), and Hamiltonian (H) at the start (*_0) and end (*_L) of trajectory
    K_0 = 0.5*(np.linalg.norm(p_0)**2)
    K_L = 0.5*(np.linalg.norm(p)**2)
    U_0 = U0    #U(z_0)
    U_L = UL    #U(z)
    H_0 = U_0 + K_0
    H_L = U_L + K_L
    # accept or reject the trajectory based on Metropolis ratio and 
    # return position vector either at the end of trajectory (z_L) or at the beginning of trajectory (z_0)
    a = np.min([1, np.exp(-H_L + H_0)])
    
    if a>np.random.uniform():   # accept
        print("Accepted!")
        count = 1
        return z, count
    else:                       # reject
        print("Rejected")
        count = 0
        return z_0, count
# ==============================================================================


z_0 = np.zeros([config.z_dim, 1])
counter = 0
counter_iter = 0
z_trace = []
for n in range(config.n_mcmc_samples):
    counter_iter += 1
    print(n)
    if n==0:
        z = z_0   
    z, count = hmc(config.hmc_epsilon, config.hmc_n_lf_steps, z, config.z_dim) 
    # counters
    z_trace.append(z)
    counter = counter + count
z_trace = np.array(z_trace).squeeze()

mcmc_post_processing(z_trace, "hmc")
