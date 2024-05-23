# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import *
from models import *
import corner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================================================================
# define the prior and likelihood objects
batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi = data_processing(
    config.z_dim, 1, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)
prior = Prior(config.prior_mean_vec, config.prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec)
likelihood = Likelihood(config.y_meas, batched_y_meas, config.like_var_vec, batched_like_var_vec)

# function to plot the unnormed posterior surface in 2D
def post_surface(z_mat):
    """
    :param z_mat: 3D array of shape (n, n, 2) containing the grid points
    :return: None
    """
    def log_post(z):
        log_prior_ = prior.log_prior_sample(z)
        log_like_ = likelihood.log_likelihood_sample(z)
        return log_prior_ + log_like_
    
    # Compute the unnormed posterior at each grid point
    log_post_mat = np.zeros([z_mat.shape[0], z_mat.shape[1]])
    for i in range(z_mat.shape[0]):
        for j in range(z_mat.shape[1]):
            log_post_mat[i, j] = log_post(z_mat[i, j, :])
    # print the sum of the unnormed posterior
    print(f"Sum of the unnormed posterior: {np.sum(np.exp(log_post_mat))}")
    # print the (i,j) index of the maximum of the unnormed posterior
    map_point = np.unravel_index(np.argmax(log_post_mat), log_post_mat.shape)
    z_map = torch.from_numpy(z_mat[map_point]).float().reshape(1, 2).to(device)
    print(z_map)
    gz_map = config.generator(z_map)
    print(f"map point: {z_map}")
    # Plot the unnormed posterior surface
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(z_mat[:, :, 0], z_mat[:, :, 1], log_post_mat, cmap="viridis")
    ax.set_xlabel(r"$z_1$", fontsize=20)
    ax.set_ylabel(r"$z_2$", fontsize=20)
    ax.set_zlabel("log_like + log_prior", fontsize=20)
    ax.set_title(r"Unnormalized Posterior Surface", fontsize=20)
    # plot the g(z_map) point
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(gz_map.detach().cpu().numpy().reshape(32, 32), cmap="gray")
    #ax.colorbar()     

    
    # Plot the contour lines of the unnormed posterior surface
    plt.figure(figsize=(10, 10))
    plt.contour(z_mat[:, :, 0], z_mat[:, :, 1], log_post_mat, 100, cmap="viridis")
    plt.xlabel(r"$z_1$", fontsize=20)
    plt.ylabel(r"$z_2$", fontsize=20)
    plt.title("log_like + log_prior", fontsize=20)
    plt.show()

# ==============================================================================
# Plot the unnormed posterior surface
#z1 = np.linspace(0.12, 1.7, 50)
#z2 = np.linspace(0.3, 1.0, 50)
z1 = np.linspace(-2, 2, 50)
z2 = np.linspace(-2, 2, 50)
z1_mat, z2_mat = np.meshgrid(z1, z2)
z_mat = np.stack([z1_mat, z2_mat], axis=2)
post_surface(z_mat)
map_point = np.argmax(z_mat[:, :, 0]).squeeze()
print(map_point.shape)
z_threshold = z_mat[map_point[0], map_point[1], :]
print(z_threshold.shape)
# threshold the z_mat to get the posterior surface
z_mat = z_mat[z_mat[:, :, 0] > (z_threshold[0]-4)]

# ==============================================================================
# Plot the corner plot showing marginal and joint distributions
# Generate samples from the posterior


