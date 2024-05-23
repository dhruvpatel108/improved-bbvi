import numpy as np
import config
from models import *


def log_var_fn_gamma(xi, batched_gamma_t):
    # gamma_t = np.exp(batched_gamma_t)
    log_var = -np.sum(batched_gamma_t, axis=1) - (0.5*np.linalg.norm(xi, axis=1)**2) - (0.5*config.x_dim*np.log(2*np.pi))
    return log_var



def bbvi_gradient(batch_size, xi, xi_orig_1d, x, mu_t, gamma_t, batched_mu_t, batched_gamma_t, prior, likelihood, if_fd_correction=False, if_cv=False):
    #grad_logq_mu = (x - batched_mu_t)/(batched_sigma_t**2)
    #grad_logq_sigma = ((x - batched_mu_t)**2/(batched_sigma_t**3)) - (1./batched_sigma_t)      
    grad_logq_mu = xi/np.exp(batched_gamma_t)
    grad_logq_gamma = xi**2 - 1.        

    log_prior_vec = np.expand_dims(prior.log_prior_batch_allterms(x), axis=1)
    log_like_vec = np.expand_dims(likelihood.log_likelihood_batch_allterms(x), axis=1)
    log_var_vec = np.expand_dims(log_var_fn_gamma(xi, batched_gamma_t), axis=1)
    grad_mu_vanilla = (grad_logq_mu.T @ (log_prior_vec + log_like_vec - log_var_vec))/batch_size
    grad_gamma_vanilla = (grad_logq_gamma.T @ (log_prior_vec + log_like_vec - log_var_vec))/batch_size

    if if_fd_correction==False and if_cv==False:
        grad_mu = grad_mu_vanilla
        grad_gamma = grad_gamma_vanilla

    if if_fd_correction==True and if_cv==False:
        grad_mu_fd, grad_gamma_fd = fd_of_log_prior_like_var(xi_orig_1d, config.epsilon, mu_t, gamma_t, prior, likelihood)    # [config.x_dim, 1] each
        first_order_correction_mu = (x - batched_mu_t) @ grad_mu_fd                     # [batch_size, 1]
        first_order_correction_gamma = (x - batched_gamma_t) @ grad_gamma_fd            # [batch_size, 1]

        grad_mu1 = (grad_logq_mu.T @ (log_prior_vec + log_like_vec - log_var_vec - first_order_correction_mu))/batch_size
        grad_gamma1 = (grad_logq_gamma.T @ (log_prior_vec + log_like_vec - log_var_vec - first_order_correction_gamma))/batch_size

        print("=====================================================")
        print(f"grad_mu1 = {grad_mu1} \n grad_mu_fd = {grad_mu_fd}")
        grad_mu = grad_mu1 + grad_mu_fd
        grad_gamma = grad_gamma1 
        print(f"grad_mu = {grad_mu} ")
    if if_fd_correction==False and if_cv==True:
        # control variate for mu
        if config.x_dim==1:
            denominator_mu = np.cov(grad_logq_mu.T)
        else:
            denominator_mu = np.diag(np.cov(grad_logq_mu.T))  
        f_mu = grad_logq_mu * (log_prior_vec + log_like_vec - log_var_vec)     # [batch_size, config.x_dim]
        #h_mu = grad_logq_mu
        expec_x2_mu = np.mean(f_mu*grad_logq_mu, axis=0)
        expec_squared_mu = np.mean(f_mu, axis=0)*np.mean(grad_logq_mu, axis=0)
        cov_fh_mu = (expec_x2_mu - expec_squared_mu)*(batch_size/(batch_size-1))
        a_cv_mu = np.expand_dims(cov_fh_mu/denominator_mu, axis=1)

        # control variate for gamma
        if config.x_dim==1:
            denominator_gamma = np.cov(grad_logq_gamma.T)
        else:
            denominator_gamma = np.diag(np.cov(grad_logq_gamma.T))
        f_gamma = grad_logq_gamma * (log_prior_vec + log_like_vec - log_var_vec)     # [batch_size, config.x_dim]
        #h_sigma = grad_logq_sigma
        expec_x2_gamma = np.mean(f_gamma*grad_logq_gamma, axis=0)
        expec_squared_gamma = np.mean(f_gamma, axis=0)*np.mean(grad_logq_gamma, axis=0)
        cov_fh_gamma = (expec_x2_gamma - expec_squared_gamma)*(batch_size/(batch_size-1))
        a_cv_gamma = np.expand_dims(cov_fh_gamma/denominator_gamma, axis=1)

        cv_correction_mu = np.expand_dims(np.mean(grad_logq_mu*a_cv_mu.T, axis=0), axis=1)
        cv_correction_gamma = np.expand_dims(np.mean(grad_logq_gamma*a_cv_gamma.T, axis=0), axis=1)

        grad_mu = grad_mu_vanilla - cv_correction_mu
        grad_gamma = grad_gamma_vanilla - cv_correction_gamma

    if if_fd_correction==True and if_cv==True:
        grad_mu_fd, grad_gamma_fd = fd_of_log_prior_like_var(xi_orig_1d, config.epsilon, mu_t, gamma_t, prior, likelihood)    # [config.x_dim, 1] each
        first_order_correction_mu = (x - batched_mu_t) @ grad_mu_fd                     # [batch_size, 1]
        first_order_correction_gamma = (x - batched_gamma_t) @ grad_gamma_fd            # [batch_size, 1]   
        grad_mu1 = (grad_logq_mu.T @ (log_prior_vec + log_like_vec - log_var_vec - first_order_correction_mu))/batch_size
        grad_gamma1 = (grad_logq_gamma.T @ (log_prior_vec + log_like_vec - log_var_vec - first_order_correction_gamma))/batch_size


        # control variate for mu
        if config.x_dim==1:
            denominator_mu = np.cov(grad_logq_mu.T)
        else:
            denominator_mu = np.diag(np.cov(grad_logq_mu.T))  
        f_mu = grad_logq_mu * (log_prior_vec + log_like_vec - log_var_vec - first_order_correction_mu)     # [batch_size, config.x_dim]
        #h_mu = grad_logq_mu
        expec_x2_mu = np.mean(f_mu*grad_logq_mu, axis=0)
        expec_squared_mu = np.mean(f_mu, axis=0)*np.mean(grad_logq_mu, axis=0)
        cov_fh_mu = (expec_x2_mu - expec_squared_mu)*(batch_size/(batch_size-1))
        a_cv_mu = np.expand_dims(cov_fh_mu/denominator_mu, axis=1)

        # control variate for gamma
        if config.x_dim==1:
            denominator_gamma = np.cov(grad_logq_gamma.T)
        else:
            denominator_gamma = np.diag(np.cov(grad_logq_gamma.T))
        f_gamma = grad_logq_gamma * (log_prior_vec + log_like_vec - log_var_vec - first_order_correction_gamma)     # [batch_size, config.x_dim]
        #h_sigma = grad_logq_sigma
        expec_x2_gamma = np.mean(f_gamma*grad_logq_gamma, axis=0)
        expec_squared_gamma = np.mean(f_gamma, axis=0)*np.mean(grad_logq_gamma, axis=0)
        cov_fh_gamma = (expec_x2_gamma - expec_squared_gamma)*(batch_size/(batch_size-1))
        a_cv_gamma = np.expand_dims(cov_fh_gamma/denominator_gamma, axis=1)

        cv_correction_mu = np.expand_dims(np.mean(grad_logq_mu*a_cv_mu.T, axis=0), axis=1)
        cv_correction_gamma = np.expand_dims(np.mean(grad_logq_gamma*a_cv_gamma.T, axis=0), axis=1)

        grad_mu = grad_mu1 - cv_correction_mu + grad_mu_fd
        grad_gamma = grad_gamma1 - cv_correction_gamma 
 
    return grad_mu, grad_gamma
    


# finite difference gradient of logq
def fd_of_log_prior_like_var(xi_orig_1d, epsilon, mu, gamma, prior, likelihood):
    # mu: [config.x_dim, 1]
    # gamma: [config.x_dim, 1]    
    gradL_mu_fd = np.zeros([config.x_dim, 1])
    for ii in range(config.x_dim):
        mu_plus = np.copy(mu)
        mu_plus[ii] = mu_plus[ii] + epsilon
        mu_minus = np.copy(mu)
        mu_minus[ii] = mu_minus[ii] - epsilon
        # x
        xn_plus = xi_orig_1d.T*np.exp(gamma) + mu_plus
        xn_minus = xi_orig_1d.T*np.exp(gamma) + mu_minus
        # prior
        log_prior_plus = prior.log_prior_sample(xn_plus)
        log_prior_minus = prior.log_prior_sample(xn_minus)
        # likelihood
        log_like_plus = likelihood.log_likelihood_sample(xn_plus)
        log_like_minus = likelihood.log_likelihood_sample(xn_minus)
        # variational
        log_var_plus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma) - (0.5*np.linalg.norm((xn_plus - mu)/np.exp(gamma))**2)
        log_var_minus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma) - (0.5*np.linalg.norm((xn_minus - mu)/np.exp(gamma))**2)
        # gradient       
        gradL_mu_fd[ii] = ((log_prior_plus + log_like_plus - log_var_plus) - (log_prior_minus + log_like_minus - log_var_minus))/(2*epsilon)


    gradL_gamma_fd = np.zeros([config.x_dim, 1])
    for ii in range(config.x_dim):
        gamma_plus = np.copy(gamma)
        gamma_plus[ii] = gamma_plus[ii] + epsilon
        gamma_minus = np.copy(gamma)
        gamma_minus[ii] = gamma_minus[ii] - epsilon
        # x
        xn_plus = xi_orig_1d.T*np.exp(gamma_plus) + mu
        xn_minus = xi_orig_1d.T*np.exp(gamma_minus) + mu
        # prior
        log_prior_plus = prior.log_prior_sample(xn_plus)
        log_prior_minus = prior.log_prior_sample(xn_minus)
        # likelihood
        log_like_plus = likelihood.log_likelihood_sample(xn_plus)
        log_like_minus = likelihood.log_likelihood_sample(xn_minus)
        # variational
        log_var_plus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma_plus) - (0.5*np.linalg.norm((xn_plus - mu)/np.exp(gamma_plus))**2)
        log_var_minus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma_minus) - (0.5*np.linalg.norm((xn_minus - mu)/np.exp(gamma_minus))**2)
        # gradient
        gradL_gamma_fd[ii] = ((log_prior_plus + log_like_plus - log_var_plus) - (log_prior_minus + log_like_minus - log_var_minus))/(2*epsilon)


    return gradL_mu_fd, gradL_gamma_fd




def fd_gradient(epsilon, xi, mu, gamma, prior, likelihood, ref_batch_size):
    """
    finite difference gradient check
    """    
    gradL_mu_fd = np.zeros([config.x_dim, 1])
    gradL_gamma_fd = np.zeros([config.x_dim, 1])
    for ii in range(config.x_dim):
        mu_plus = np.copy(mu)
        mu_plus[ii] = mu_plus[ii] + epsilon
        mu_minus = np.copy(mu)
        mu_minus[ii] = mu_minus[ii] - epsilon
    
        mu_plus_2d = np.repeat(mu_plus.T, ref_batch_size, axis=0)
        mu_minus_2d = np.repeat(mu_minus.T, ref_batch_size, axis=0)
        gamma_2d = np.repeat(gamma.T, ref_batch_size, axis=0)
        xx_plus = xi*np.exp(gamma_2d) + mu_plus_2d
        xx_minus = xi*np.exp(gamma_2d) + mu_minus_2d
    
        L_batch_plus = 0.
        L_batch_minus = 0.
        for n in range(ref_batch_size):
            xn_plus = np.expand_dims(xx_plus[n, :], axis=1)
            xn_minus = np.expand_dims(xx_minus[n, :], axis=1)
            # prior
            log_prior_plus = prior.log_prior_sample(xn_plus)
            log_prior_minus = prior.log_prior_sample(xn_minus)
            # likelihood
            log_like_plus = likelihood.log_likelihood_sample(xn_plus)
            log_like_minus = likelihood.log_likelihood_sample(xn_minus)
            # variational
            log_var_plus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma) - (0.5*np.linalg.norm((xn_plus - mu)/np.exp(gamma))**2)
            log_var_minus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma) - (0.5*np.linalg.norm((xn_minus - mu)/np.exp(gamma))**2)
            # ELBO
            L_batch_plus = L_batch_plus + (log_prior_plus + log_like_plus - log_var_plus)
            L_batch_minus = L_batch_minus + (log_prior_minus + log_like_minus - log_var_minus)
        
        elbo_plus = L_batch_plus/ref_batch_size
        elbo_minus = L_batch_minus/ref_batch_size
        gradL_mu_fd[ii] = (elbo_plus - elbo_minus)/(2*epsilon)


    for ii in range(config.x_dim):
        gamma_plus = np.copy(gamma)
        gamma_plus[ii] = gamma_plus[ii] + epsilon
        gamma_minus = np.copy(gamma)
        gamma_minus[ii] = gamma_minus[ii] - epsilon
    
        mu_2d = np.repeat(mu.T, ref_batch_size, axis=0)
        gamma_plus_2d = np.repeat(gamma_plus.T, ref_batch_size, axis=0)
        gamma_minus_2d = np.repeat(gamma_minus.T, ref_batch_size, axis=0)
        xx_plus = xi*np.exp(gamma_plus_2d) + mu_2d
        xx_minus = xi*np.exp(gamma_minus_2d) + mu_2d
    
        L_batch_plus = 0.
        L_batch_minus = 0.
        for n in range(ref_batch_size):
            xn_plus = np.expand_dims(xx_plus[n, :], axis=1)
            xn_minus = np.expand_dims(xx_minus[n, :], axis=1)
            # prior
            log_prior_plus = prior.log_prior_sample(xn_plus)
            log_prior_minus = prior.log_prior_sample(xn_minus)
            # likelihood
            log_like_plus = likelihood.log_likelihood_sample(xn_plus)
            log_like_minus = likelihood.log_likelihood_sample(xn_minus)
            # variational
            log_var_plus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma_plus) - (0.5*np.linalg.norm((xn_plus - mu)/np.exp(gamma_plus))**2)
            log_var_minus = -(0.5*config.x_dim*np.log(2*np.pi)) - np.sum(gamma_minus) - (0.5*np.linalg.norm((xn_minus - mu)/np.exp(gamma_minus))**2)
            # ELBO
            L_batch_plus = L_batch_plus + (log_prior_plus + log_like_plus - log_var_plus)
            L_batch_minus = L_batch_minus + (log_prior_minus + log_like_minus - log_var_minus)
        
        elbo_plus = L_batch_plus/ref_batch_size
        elbo_minus = L_batch_minus/ref_batch_size
        gradL_gamma_fd[ii] = (elbo_plus - elbo_minus)/(2*epsilon)


    return gradL_mu_fd, gradL_gamma_fd