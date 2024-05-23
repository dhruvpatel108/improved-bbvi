import numpy as np
from utils import *
import config

class Prior():
    """Prior distribution for z.
    """
    def __init__(self, prior_mean_vec, prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec):
        self.prior_mean_vec = prior_mean_vec
        self.prior_var_vec = prior_var_vec
        self.batched_prior_mean_vec = batched_prior_mean_vec
        self.batched_prior_var_vec = batched_prior_var_vec

    def log_prior_batch(self, z):
        batched_prior_std_vec = np.sqrt(self.batched_prior_var_vec)
        log_prior = - (np.sum(np.log(batched_prior_std_vec), axis=1)) - (0.5*(np.linalg.norm((z - self.batched_prior_mean_vec)/batched_prior_std_vec, axis=1)**2)) #- (0.5*self.batched_prior_mean_vec.shape[1]*np.log(2*np.pi)) 
        return log_prior
    
    def log_prior_batch_allterms(self, z):
        batched_prior_std_vec = np.sqrt(self.batched_prior_var_vec)
        log_prior = - (0.5*self.batched_prior_mean_vec.shape[1]*np.log(2*np.pi)) - (np.sum(np.log(batched_prior_std_vec), axis=1)) - (0.5*(np.linalg.norm((z - self.batched_prior_mean_vec)/batched_prior_std_vec, axis=1)**2)) 
        return log_prior

    def log_prior_sample(self, z):
        prior_std_vec = np.sqrt(self.prior_var_vec)
        log_prior_sample = - (0.5*self.prior_mean_vec.shape[0]*np.log(2*np.pi)) - (np.sum(np.log(prior_std_vec))) - (0.5*(np.linalg.norm((z - self.prior_mean_vec)/prior_std_vec)**2)) 
        return log_prior_sample


class Prior_Xi():
    """Prior distribution for xi sampled from a 4-dimensional Uniform distribution
    xi1 ~ U(0.2*L, 0.4*L)
    xi2 ~ U(0.2*L, 0.4*L)
    xi3 ~ U(0.6*L, 0.8*L)
    xi4 ~ U(0.6*L, 0.8*L)
    L = 2*np.pi
    """
    def __init__(self, xi1_min, xi1_max, xi2_min, xi2_max, xi3_min, xi3_max, xi4_min, xi4_max):
        self.xi1_min = xi1_min
        self.xi1_max = xi1_max
        self.xi2_min = xi2_min
        self.xi2_max = xi2_max
        self.xi3_min = xi3_min
        self.xi3_max = xi3_max
        self.xi4_min = xi4_min
        self.xi4_max = xi4_max

    def log_prior_sample(self, xi):
        #prior_sample = 1/((self.xi1_max - self.xi1_min)*(self.xi2_max - self.xi2_min)*(self.xi3_max - self.xi3_min)*(self.xi4_max - self.xi4_min))
        #log_prior_sample = np.log(prior_sample)
        #log_prior_sample = -np.log((self.xi1_max - self.xi1_min)) - np.log((self.xi2_max - self.xi2_min)) 
        #- np.log((self.xi3_max - self.xi3_min)) - np.log((self.xi4_max - self.xi4_min))
        return log_prior_sample
    
        


class Likelihood():
    def __init__(self, y_meas, batched_y_meas, like_var_vec, batched_like_var_vec, batch_size):
        self.y_meas = y_meas
        self.batched_y_meas = batched_y_meas
        self.like_var_vec = like_var_vec
        self.batched_like_var = batched_like_var_vec
        self.batch_size = batch_size

    def log_likelihood_batch(self, z):
        batched_like_std_vec = np.sqrt(self.batched_like_var)
        batched_y_pred = pred_y_from_z(z, config.generator, config.F, config.min_value, config.max_value, self.batch_size, config.x_dim, config.like_dim)
        res = self.batched_y_meas - batched_y_pred
        log_like = - (np.sum(np.log(batched_like_std_vec), axis=1)) - ((0.5*np.linalg.norm(res/batched_like_std_vec, axis=1)**2)) #- (0.5*self.batched_y_meas.shape[1]*np.log(2*np.pi))
        return log_like
    
    def log_likelihood_batch_allterms(self, z):
        batched_like_std_vec = np.sqrt(self.batched_like_var)
        batched_y_pred = pred_y_from_z(z, config.generator, config.F, config.min_value, config.max_value, self.batch_size, config.x_dim, config.like_dim)
        res = self.batched_y_meas - batched_y_pred
        log_like = - (0.5*self.batched_y_meas.shape[1]*np.log(2*np.pi)) - (np.sum(np.log(batched_like_std_vec), axis=1)) - ((0.5*np.linalg.norm(res/batched_like_std_vec, axis=1)**2))
        return log_like

    def log_likelihood_sample(self, z):
        like_std_vec = np.sqrt(self.like_var_vec)
        y_pred = pred_y_from_z(z, config.generator, config.F, config.min_value, config.max_value, 1, config.x_dim, config.like_dim)
        res = self.y_meas - y_pred.T
        log_like_sample = - (0.5*self.y_meas.shape[0]*np.log(2*np.pi))  - (np.sum(np.log(like_std_vec))) - (0.5*np.linalg.norm(res/like_std_vec)**2) 
        return log_like_sample


"""
class Variational():
    def __init__(self, variational_mean_vec, variational_var_vec, batched_variational_mean_vec, batched_variational_var_vec):
        self.variational_mean_vec = variational_mean_vec
        self.variational_var_vec = variational_var_vec
        self.batched_variational_mean_vec = batched_variational_mean_vec
        self.batched_variational_var_vec = batched_variational_var_vec

    def log_variational_batch(self, x):
        batched_variational_std_vec = np.sqrt(self.batched_variational_var_vec)
        log_variational = - (0.5*self.batched_variational_mean_vec.shape[1]*np.log(2*np.pi)) - (np.sum(np.log(batched_variational_std_vec), axis=1)) - (0.5*(np.linalg.norm((x - self.batched_variational_mean_vec)/batched_variational_std_vec, axis=1)**2))
        return log_variational

    def log_variational_batch_grad_mu_sigma(self, x):
        batched_variational_std_vec = np.sqrt(self.batched_variational_var_vec)
        grad_logq_mu = (x - self.batched_variational_mean_vec)/self.batched_variational_var_vec
        grad_logq_sigma = (((x - self.batched_variational_mean_vec)**2)/(batched_variational_std_vec**3)) - (1./batched_variational_std_vec)
        return grad_logq_mu, grad_logq_sigma

    def log_variational_sample(self, x):
        variational_std_vec = np.sqrt(self.variational_var_vec)
        log_variational_sample = - (0.5*self.variational_mean_vec.shape[0]*np.log(2*np.pi)) - (np.sum(np.log(variational_std_vec))) - (0.5*(np.linalg.norm((x - self.variational_mean_vec)/variational_std_vec)**2))
        return log_variational_sample
"""

