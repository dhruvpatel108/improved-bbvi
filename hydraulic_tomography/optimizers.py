import config
import numpy as np
from scipy.optimize import minimize
from gradient import bbvi_gradient
from utils import *

# ==================== OPTIMIZATION ====================
# gradient descent
def gradient_descent(xi, xi_orig_1d, mu_init, gamma_init, n_iter, mu_step_size, gamma_step_size, prior, likelihood, prior_elbo, likelihood_elbo,
                     if_fd_correction, if_cv):
    mu_list = []
    gamma_list = []
    elbo_list = []
    mu_t = np.copy(mu_init)
    gamma_t = np.copy(gamma_init)
    #velocity_mu = np.zeros([x_dim, 1])
    #velocity_sigma = np.zeros([x_dim, 1])  
    for t in range(n_iter):
        mu_list.append(mu_t)
        gamma_list.append(gamma_t)
        batched_mu_t = np.repeat(mu_t.T, config.batch_size, axis=0) 
        batched_gamma_t = np.repeat(gamma_t.T, config.batch_size, axis=0)
        batched_xx_t = xi*np.exp(batched_gamma_t) + batched_mu_t
        # Done till here...!
        bbvi_grad_mu, bbvi_grad_gamma = bbvi_gradient(xi, xi_orig_1d, batched_xx_t, mu_t, gamma_t, batched_mu_t, batched_gamma_t, prior, likelihood, if_fd_correction, if_cv)
        #velocity_mu = 0.9*velocity_mu + 0.1*bbvi_grad_mu
        #velocity_sigma = 0.9*velocity_sigma + 0.1*bbvi_grad_sigma
        #mu_t = mu_t + mu_step_size*velocity_mu
        #sigma_t = sigma_t + sigma_step_size*velocity_sigma
        mu_t = mu_t + mu_step_size*bbvi_grad_mu
        gamma_t = gamma_t + gamma_step_size*bbvi_grad_gamma
        elbo_list.append(ELBO_batch(mu_t, np.exp(gamma_t), prior_elbo, likelihood_elbo, config.elbo_batch_size))  
        print(f"iter = {t}, ELBO = {elbo_list[-1]:.4f}, mu_t[0] = {mu_t[0, 0]:.4f}, mu_t[1] = {mu_t[1, 0]:.4f}, gamma_t[0] = {gamma_t[0, 0]:.4f}, gamma_t[1] = {gamma_t[1, 0]:.4f}")

        
    return mu_t, gamma_t, np.array(mu_list).squeeze(), np.array(gamma_list).squeeze(), elbo_list


# adam
def adam(xi, xi_orig_1d, mu_init, gamma_init, n_iter, mu_step_size, gamma_step_size, prior, likelihood, prior_elbo, likelihood_elbo,
         if_fd_correction, if_cv, b1_mu=0.8, b2_mu=0.8, b1_gamma=0.8, b2_gamma=0.8, eps=1e-8):
    mu_list = []
    gamma_list = []
    elbo_list = []
    mu_t = np.copy(mu_init)
    gamma_t = np.copy(gamma_init)
    m_mu = np.zeros_like(mu_t)
    m_gamma = np.zeros_like(gamma_t)
    v_mu = np.zeros_like(mu_t)
    v_gamma = np.zeros_like(gamma_t)
    for t in range(n_iter):
        mu_list.append(mu_t)
        gamma_list.append(gamma_t)
        batched_mu_t = np.repeat(mu_t.T, config.batch_size, axis=0)
        batched_gamma_t = np.repeat(gamma_t.T, config.batch_size, axis=0)
        batched_xx_t = xi*np.exp(batched_gamma_t) + batched_mu_t
        bbvi_grad_mu, bbvi_grad_gamma = bbvi_gradient(xi, xi_orig_1d, batched_xx_t, mu_t, gamma_t, batched_mu_t, batched_gamma_t, prior, likelihood, if_fd_correction, if_cv)
        # update per-componentlearning rate scaling (g)
        m_mu = b1_mu*m_mu + (1-b1_mu)*bbvi_grad_mu
        m_gamma = b1_gamma*m_gamma + (1-b1_gamma)*bbvi_grad_gamma
        v_mu = b2_mu*v_mu + (1-b2_mu)*bbvi_grad_mu**2
        v_gamma = b2_gamma*v_gamma + (1-b2_gamma)*(bbvi_grad_gamma**2)
        # bias correction
        m_mu_hat = m_mu/(1-b1_mu**(t+1))
        m_gamma_hat = m_gamma/(1-b1_gamma**(t+1))
        v_mu_hat = v_mu/(1-b2_mu**(t+1))
        v_gamma_hat = v_gamma/(1-b2_gamma**(t+1))

        # gradient ascent
        mu_t = mu_t + mu_step_size*m_mu_hat/(np.sqrt(v_mu_hat) + eps)
        gamma_t = gamma_t + gamma_step_size*m_gamma_hat/(np.sqrt(v_gamma_hat) + eps)  
        elbo_list.append(ELBO_batch(mu_t, np.exp(gamma_t), prior_elbo, likelihood_elbo, config.elbo_batch_size))  
        print(f"iter = {t}, ELBO = {elbo_list[-1]:.4f}, mu_t[0] = {mu_t[0, 0]:.4f}, mu_t[1] = {mu_t[1, 0]:.4f}, gamma_t[0] = {gamma_t[0, 0]:.4f}, gamma_t[1] = {gamma_t[1, 0]:.4f}")

    return mu_t, gamma_t, np.array(mu_list).squeeze(), np.array(gamma_list).squeeze(), elbo_list




"""
# adagrad
def adagrad(xi, xi_orig_1d, mu_init, sigma_init, n_iter, mu_step_size, sigma_step_size, prior, likelihood, if_fd_correction, if_cv, epsilon=1e-7):
    mu_list = []
    sigma_list = []
    mu_t = np.copy(mu_init)
    sigma_t = np.copy(sigma_init)
    g_mu = np.zeros_like(mu_t)
    g_sigma = np.zeros_like(sigma_t)
    for t in range(n_iter):
        mu_list.append(mu_t)
        sigma_list.append(sigma_t)
        batched_mu_t = np.repeat(mu_t.T, config.batch_size, axis=0)
        batched_sigma_t = np.repeat(sigma_t.T, config.batch_size, axis=0)
        batched_xx_t = xi*batched_sigma_t + batched_mu_t
        bbvi_grad_mu, bbvi_grad_sigma = bbvi_gradient(xi, xi_orig_1d, batched_xx_t, mu_t, sigma_t, batched_mu_t, batched_sigma_t, prior, likelihood, if_fd_correction, if_cv)
        # update per-componentlearning rate scaling (g)
        g_mu += bbvi_grad_mu ** 2
        g_sigma += bbvi_grad_sigma ** 2
        # gradient ascent
        mu_t = mu_t + mu_step_size*bbvi_grad_mu/np.sqrt(g_mu + epsilon)
        sigma_t = sigma_t + sigma_step_size*bbvi_grad_sigma/np.sqrt(g_sigma + epsilon)
        print(f"iter = {t}, mu_t[0] = {mu_t[0, 0]:.4f}, mu_t[1] = {mu_t[1, 0]:.4f}, sigma_t[0] = {sigma_t[0, 0]:.4f}, sigma_t[1] = {sigma_t[1, 0]:.4f}")

    return mu_t, sigma_t, np.array(mu_list).squeeze(), np.array(sigma_list).squeeze()





# adam with coordinate ascent
def adam_coordinate_ascent(xi, xi_orig_1d, mu_init, sigma_init, n_iter, mu_step_size, sigma_step_size, prior, likelihood, if_fd_correction, if_cv, 
                           b1_mu=0.8, b2_mu=0.8, b1_sigma=0.5, b2_sigma=0.5, eps=1e-8):
    mu_list = []
    sigma_list = []
    mu_t = np.copy(mu_init)
    sigma_t = np.copy(sigma_init)
    m_mu = np.zeros_like(mu_t)
    m_sigma = np.zeros_like(sigma_t)
    v_mu = np.zeros_like(mu_t)
    v_sigma = np.zeros_like(sigma_t)
    for t in range(n_iter):
        mu_list.append(mu_t)
        sigma_list.append(sigma_t)
        batched_mu_t = np.repeat(mu_t.T, config.batch_size, axis=0)
        batched_sigma_t = np.repeat(sigma_t.T, config.batch_size, axis=0)
        batched_xx_t = xi*batched_sigma_t + batched_mu_t
        bbvi_grad_mu, bbvi_grad_sigma = bbvi_gradient(xi, xi_orig_1d, batched_xx_t, mu_t, sigma_t, batched_mu_t, batched_sigma_t, prior, likelihood, if_fd_correction, if_cv)
        # update per-componentlearning rate scaling (g)
        # coordinate ascent
        for dd in range(mu_t.shape[0]):
            m_mu[dd, 0] = b1_mu*m_mu[dd, 0] + (1-b1_mu)*bbvi_grad_mu[dd, 0]
            m_sigma[dd, 0] = b1_sigma*m_sigma[dd, 0] + (1-b1_sigma)*bbvi_grad_sigma[dd, 0]
            v_mu[dd, 0] = b2_mu*v_mu[dd, 0] + (1-b2_mu)*bbvi_grad_mu[dd, 0]**2
            v_sigma[dd, 0] = b2_sigma*v_sigma[dd, 0] + (1-b2_sigma)*(bbvi_grad_sigma[dd, 0]**2)
            # bias correction
            m_mu_hat = m_mu[dd, 0]/(1-b1_mu**(t+1))
            m_sigma_hat = m_sigma[dd, 0]/(1-b1_sigma**(t+1))
            v_mu_hat = v_mu[dd, 0]/(1-b2_mu**(t+1))
            v_sigma_hat = v_sigma[dd, 0]/(1-b2_sigma**(t+1))

            # gradient ascent
            mu_t[dd, 0] = mu_t[dd, 0] + mu_step_size*m_mu_hat/(np.sqrt(v_mu_hat) + eps)
            sigma_t[dd, 0] = sigma_t[dd, 0] + sigma_step_size*m_sigma_hat/(np.sqrt(v_sigma_hat) + eps)
        
        print(f"iter = {t}, mu_t[0] = {mu_t[0, 0]:.4f}, mu_t[1] = {mu_t[1, 0]:.4f}, sigma_t[0] = {sigma_t[0, 0]:.4f}, sigma_t[1] = {sigma_t[1, 0]:.4f}")

    return mu_t, sigma_t, np.array(mu_list).squeeze(), np.array(sigma_list).squeeze()

"""

"""
def f_and_jac(mu_sigma, info):
    print(f"mu_sigma.shape={mu_sigma.shape}")
    #print(f"mu.shape={mu.shape}, sigma.shape={sigma.shape}, xi.shape={xi.shape}")
    #batched_mu = np.repeat(np.expand_dims(mu, axis=1).T, batch_size, axis=0)
    #batched_sigma = np.repeat(np.expand_dims(sigma, axis=1).T, batch_size, axis=0)
    mu = np.expand_dims(mu_sigma[:x_dim], axis=1)
    sigma = np.expand_dims(mu_sigma[x_dim:], axis=1)
    batched_mu = np.repeat(mu.T, batch_size, axis=0)
    batched_sigma = np.repeat(sigma.T, batch_size, axis=0)
    #print(f"mu_2d.shape={mu_2d.shape}, xi*sigma.shape={(xi*sigma).shape}")
    #print(f"xi.shape={xi.shape}, batched_sigma.shape={batched_sigma.shape}, batched_mu.shape={batched_mu.shape}")
    zz = xi*batched_sigma + batched_mu
    L_batch = 0.
    for n in range(elbo_batch_size):
        zn = np.expand_dims(zz[n, :], axis=1)
        # prior
        log_prior_n = prior.log_prior_sample(zn)#-((x_dim/2)*np.log(2*np.pi)) - (0.5*(np.linalg.norm(zn)**2))
        # likelihood            
        #u_pred = F @ mu #gz_int
        #diff_np = (u_pred - u_meas)
        #log_like_n = -((like_dim/2)*np.log(2*np.pi)) - (like_dim*np.log(like_stddev)) - (np.linalg.norm(diff_np)**2/(2*like_var))
        #print(f"zn.shape={zn.shape}, mu.shape={mu.shape}, sigma.shape={sigma.shape}")

        log_like_n = likelihood.log_likelihood_sample(zn)
        # variational 
        #log_var_n = -((z_dim/2)*np.log(2*np.pi)) - (z_dim*np.log(sigma)) - ((np.linalg.norm(zn - mu)**2)/(2*(sigma**2)))        
        log_var_n = -(0.5*x_dim*np.log(2*np.pi)) - np.sum(np.log(sigma)) - (0.5*np.linalg.norm((zn - mu)/sigma)**2)
        # elbo contribution
        Ln = log_like_n + log_prior_n - log_var_n
        L_batch = L_batch + Ln

    # gradient calculation
    jac_mu, jac_sigma = bbvi_gradient(xi, zz, mu, sigma, batched_mu, batched_sigma, if_fd_correction=False, if_cv=False)
    #jac = bbvi_with_cv_gradient(zz, np.expand_dims(mu, axis=1), mu_2d).squeeze()
    #print(f"jac_mu.shape={jac_mu.shape}, jac_sigma.shape={jac_sigma.shape}")
    jac = np.concatenate((jac_mu, jac_sigma), axis=0).squeeze()
    #print(f"jac.shape={jac.shape}")

    # logging info
    if (info["Nfeval"]%1)==0:
        info["fun_val"].append(L_batch)
        info["mu_norm"].append(np.linalg.norm(mu))
        info["sigma_norm"].append(np.linalg.norm(sigma))
        info["grad_norm"].append(np.linalg.norm(jac))
        print(f"iter={info['Nfeval']}  |  fun_val = {L_batch} | mu[0] = {mu[0]} | mu[1] = {mu[1]}, sigma[0] = {sigma[0]}, sigma[1] = {sigma[1]}")
    info["Nfeval"] += 1 

    return -L_batch, -jac



elbo_batch_size = batch_size
info={}
info["Nfeval"] = 0
info["fun_val"] = []
info["mu_norm"] = []
info["sigma_norm"] = []
info["grad_norm"] = []
print(f"mu_init_guess.shape={mu_init_guess.shape}, sigma_init_guess.shape={sigma_init_guess.shape}, mu_sigma_init_guess.shape={np.concatenate((mu_init_guess, sigma_init_guess), axis=0).shape}")
f_and_jac(np.concatenate((mu_init_guess, sigma_init_guess), axis=0).squeeze(), info)
print(info)
res = minimize(f_and_jac, np.concatenate((mu_init_guess, sigma_init_guess), axis=0).squeeze(), method='BFGS', jac=True, args=(info,),
               options={'disp': True, 'maxiter':max_iter, 'gtol':1e-6*10000})
#mu_opt = res.x
mu_sigma_opt = res.x
mu_opt = mu_sigma_opt[:x_dim]
sigma_opt = mu_sigma_opt[x_dim:]
"""

