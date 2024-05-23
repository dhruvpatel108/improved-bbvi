import yaml
import torch
import numpy as np
#import config
from models_sagan import Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================
# Problem parameters 
with open("params.yml", 'r') as config_file:
    config = yaml.safe_load(config_file)

seed_no = config["seed_no"]
np.random.seed(seed_no)
torch.manual_seed(seed_no)
torch.cuda.manual_seed(seed_no)
# physical parameters
min_value = config["min_value"]
max_value = config["max_value"]
x_dim = config["img_size"]
test_img_index = config["test_img_index"]
noise_var = config["noise_var"]
like_var = config["like_var"]
k = config["k"]
# GAN parameters
z_dim = config["latent_dim"]
# VI parameters
xi_sampling = config["xi_sampling"]
optimizer = config["optimizer"]
n_iter = config["n_iter"]
mu_step_size = config["mu_step_size"]
gamma_step_size = config["gamma_step_size"]
batch_size = config["batch_size"]
elbo_batch_size = config["elbo_batch_size"]
mu_init_scalar = config["mu_init_scalar"]
sigma_init_scalar = config["sigma_init_scalar"]
epsilon = config["epsilon"]
mu_init_guess = np.random.randn(z_dim, 1)*0.01 + mu_init_scalar                       #np.ones([z_dim, 1])*mu_init_scalar
gamma_init_guess = np.random.randn(z_dim, 1)*0.01 + sigma_init_scalar                 #np.log(np.ones([z_dim, 1])*sigma_init_scalar)
# posterior parameters
n_samples = config["n_samples"]
# MCMC parameters
n_mcmc_samples = config["n_mcmc_samples"]
mcmc_prop_var = config["mcmc_prop_var"]
burn_in = config["burn_in"]
hmc_epsilon = config["hmc_epsilon"]
hmc_n_lf_steps = config["hmc_n_lf_steps"]
# =============================================================================


# =============================================================================
# prior
prior_mean_vec = np.zeros([z_dim, 1])
prior_var_vec = np.ones([z_dim, 1])*1.0
#cov_z = np.eye(z_dim)*prior_var_vec
# likelihood
like_dim = (x_dim-2)**2
like_var_vec = np.ones([like_dim, 1])*like_var
#cov_y = np.eye(like_dim)*like_var_vec
# =============================================================================


# =============================================================================
# data
data_dir = f"data/testimgindex{test_img_index}_k{k}_noisevar{noise_var}"
x_true = np.load(f"{data_dir}/x_true.npy").reshape([x_dim, x_dim])
y_true = np.load(f"{data_dir}/u_true.npy").reshape([like_dim, 1])
y_meas = np.load(f"{data_dir}/u_noisy.npy").reshape([like_dim, 1])
print(f"y_meas.shape = {y_meas.shape}, x_true.shape = {x_true.shape}, y_true.shape = {y_true.shape}")
F = np.load(f"{data_dir}/F.npy")
Ft = torch.from_numpy(F).float()
y_meas_torch = torch.from_numpy(y_meas).float()
# =============================================================================


# =============================================================================
# GAN
chkpt_path = f"./checkpoint/model_zdim{z_dim}.pth"
# load checkpoint
def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['G_state_dict'])
    return model

# model
generator = Generator(image_size=x_dim, z_dim=z_dim, batch_size=1).to(device)
# load checkpoint
generator = load_checkpoint(chkpt_path, generator)
# =============================================================================