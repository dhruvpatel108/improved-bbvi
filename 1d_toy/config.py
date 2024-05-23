import yaml
import numpy as np
import config
# =============================================================================
# Problem parameters 
with open("params.yml", 'r') as config_file:
    config = yaml.safe_load(config_file)
    
optimizer = config["optimizer"]
x_dim = config["x_dim"]
n_iter = config["n_iter"]
mu_step_size = config["mu_step_size"]
gamma_step_size = config["gamma_step_size"]
batch_size = config["batch_size"]
mu_init_scalar = config["mu_init_scalar"]
gamma_init_scalar = config["gamma_init_scalar"]
epsilon = config["epsilon"]
#gamma = config["gamma"]
mu_init_guess = np.ones([x_dim, 1])*mu_init_scalar
gamma_init_guess = np.log(np.ones([x_dim, 1])*gamma_init_scalar)
# =============================================================================




# =============================================================================
# prior
prior_mean_vec = np.ones([x_dim, 1])*2.0 
prior_var_vec = np.linspace(1, 1, x_dim).reshape([x_dim, 1])
cov_x = np.eye(x_dim)*prior_var_vec
# likelihood
#noise_var = 1.0
#like_var = noise_var
noise_var_vec = np.linspace(1, 1, x_dim).reshape([x_dim, 1])
like_var_vec = noise_var_vec
cov_y = np.eye(x_dim)*noise_var_vec
y_meas = np.ones([x_dim, 1])*2.0
# =============================================================================