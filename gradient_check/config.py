import yaml
import torch
from PIL import Image
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchinfo import summary
# =============================================================================
# Problem parameters 
with open("params.yml", 'r') as config_file:
    config = yaml.safe_load(config_file)

seed_no = config["seed_no"]
np.random.seed(seed_no)
torch.manual_seed(seed_no)
torch.cuda.manual_seed(seed_no)
# =============================================================================
# Problem parameters
x_dim = config["x_dim"]
# VI parameters
xi_sampling = config["xi_sampling"]
ref_batch_size = config["ref_batch_size"]
mu_init_scalar = config["mu_init_scalar"]
sigma_init_scalar = config["sigma_init_scalar"]
epsilon = config["epsilon"]
#mu_init_guess = np.ones([x_dim, 1])*mu_init_scalar
#gamma_init_guess = np.log(np.ones([x_dim, 1])*sigma_init_scalar)
np_save = config["np_save"]


# =============================================================================
# prior
prior_mean_vec = np.ones([x_dim, 1])*10.0
prior_var_vec = np.linspace(1, 1, x_dim).reshape([x_dim, 1])
cov_x = np.eye(x_dim)*prior_var_vec
# likelihood
like_var_vec = np.linspace(1, 1, x_dim).reshape([x_dim, 1])
cov_y = np.eye(x_dim)*like_var_vec
A_scalar = 1.
A = np.eye(x_dim)*A_scalar
y_meas = np.zeros([x_dim, 1])
# true posterior stats.
true_mu_post = prior_mean_vec + (((cov_x @ A.T) @ np.linalg.inv(cov_y + (A@cov_x@A.T))) @ (y_meas - (A@prior_mean_vec)))
cov_post = cov_x - (((cov_x @ A.T) @ np.linalg.inv(cov_y + (A@cov_x@A.T))) @ (A @ cov_x))
true_std_post = np.sqrt(np.diag(cov_post))
print(f"true_mu_post = {true_mu_post} \n cov_post = {cov_post} \n true_std_post = {true_std_post}")
true_post_samples = np.random.multivariate_normal(mean=true_mu_post.squeeze(), cov=cov_post, size=1000)
# =============================================================================


