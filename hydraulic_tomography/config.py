import yaml
import torch
from PIL import Image
import numpy as np
from models_gan import Generator
from models_cnn_allSourceTogether import CNN_Surrogate
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
# physical parameters
min_value = config["min_value"]
max_value = config["max_value"]
box_z_dim = config["box_z_dim"]
box_x_dim = config["box_x_dim"]
like_var = config["like_var"]
n_pressure_sensors = config["n_pressure_sensors"]
n_source_sensors = config["n_source_sensors"]
# GAN parameters
z_dim = config["latent_dim"]
gan_dim = config["gan_dim"]
# CNN parameters
cnn_dim = config["cnn_dim"]
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
like_dim = n_pressure_sensors*n_source_sensors
like_var_vec = np.ones([like_dim, 1])*like_var
#cov_y = np.eye(like_dim)*like_var_vec
# =============================================================================


# =============================================================================
# data
ktrue = np.load("./gan/data/true-1610x756.npy").astype(np.float32)
source = np.load("./experimental_dataprocessing/source_array.npy")
obs = np.load("./experimental_dataprocessing/obs_array.npy")
bc = np.load("./experimental_dataprocessing/bc_array.npy")
y_meas = obs[:, 2].reshape([like_dim, 1])
#data_dir = f"data/testimgindex{test_img_index}_k{k}_noisevar{noise_var}"
#x_true = np.load(f"{data_dir}/x_true.npy").reshape([x_dim, x_dim])
#y_true = np.load(f"{data_dir}/u_true.npy").reshape([like_dim, 1])
#y_meas = np.load(f"{data_dir}/u_noisy.npy").reshape([like_dim, 1])
# true
ktrue[ktrue==0] = 0.26
ktrue[ktrue==1] = 0.0161
ktrue[ktrue==2] = 0.0199
ktrue[ktrue==3] = 0.0642
ktrue_ = np.array(Image.fromarray(ktrue).resize((box_x_dim, box_z_dim), Image.NEAREST))
x_true = np.flipud(ktrue_)



print(f"y_meas.shape = {y_meas.shape}, x_true.shape = {x_true.shape}")
y_meas_torch = torch.from_numpy(y_meas).float()
# =============================================================================


# =============================================================================
# Load generator and surrogate forward model
ckpt_gen = torch.load("./2stage_hmc/ckpt/gen_9999.pt")
ckpt_cnn = torch.load("./2stage_hmc/ckpt/dim4_channel1_model10000.pth")
# generator
generator = Generator(img_size=(box_z_dim, box_x_dim, 1), latent_dim=z_dim, dim=gan_dim).to(device)
generator.load_state_dict(ckpt_gen)
# surrogate
cnn = CNN_Surrogate(dim=cnn_dim).to(device)
cnn.load_state_dict(ckpt_cnn)
# =============================================================================
# print summary
#print("Generator summary:")
#summary(generator, input_size=(batch_size, z_dim))
#print("CNN summary:")
#summary(cnn, input_size=(batch_size, 1, box_z_dim, box_x_dim))

