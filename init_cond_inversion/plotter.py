# import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config
plt.rcParams.update({'font.size': 16})
# ==============================================================================
# load data
data_path = "./data/testimgindex0_k0.064_noisevar1.0"
bbvi_path = "./exps/id0_k0.064_noisevar1.0_likevar1.0/Zdim5/bbvi/xi_normal_symmetric/adam/BS150_N1000_seed5"
mcmc_path = "./exps/id0_k0.064_noisevar1.0_likevar1.0/Zdim5/mcmc/N10000_PropVar0.01"
x_true = np.load(f"{data_path}/x_true.npy")
y_true = np.load(f"{data_path}/u_true.npy")
y_meas = np.load(f"{data_path}/u_noisy.npy")
#--
x_mean_bbvi = np.load(f"{bbvi_path}/x_mean.npy")
x_std_bbvi = np.load(f"{bbvi_path}/x_std.npy")
#--
x_mean_mcmc = np.load(f"{mcmc_path}/x_mean.npy")
x_std_mcmc = np.load(f"{mcmc_path}/x_std.npy")



# function to plot the posterior statistics
def post_stats_plots(x_true, y_true, y_meas, x_mean_bbvi, x_std_bbvi, x_mean_mcmc, x_std_mcmc):
    cmap = "viridis"
    fig, ax = plt.subplots(3, 3, figsize=(14, 17))
    im = ax[0, 0].imshow(x_true, cmap=cmap)
    ax[0, 0].set_title("True")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    im = ax[0, 1].imshow(y_true.reshape([config.x_dim-2, config.x_dim-2]), cmap=cmap)
    ax[0, 1].set_title("Measurement (w/out noise)")
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    im = ax[0, 2].imshow(y_meas.reshape([config.x_dim-2, config.x_dim-2]), cmap=cmap)
    ax[0, 2].set_title("Measurement")
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    #im = ax[1, 0].imshow(x_map, cmap=cmap)
    #ax[1, 0].set_title("MAP")
    #ax[1, 0].set_xticks([])
    #ax[1, 0].set_yticks([])
    #divider = make_axes_locatable(ax[1, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im, cax=cax, orientation="vertical")
    im = ax[1, 1].imshow(x_mean_bbvi, cmap=cmap)
    ax[1, 1].set_title("Mean (BBVI)")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    im = ax[1, 2].imshow(x_std_bbvi, cmap=cmap)
    ax[1, 2].set_title("Standard Deviation (BBVI)")
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    # MCMC
    im = ax[2, 1].imshow(x_mean_mcmc, cmap=cmap)
    ax[2, 1].set_title("Mean (MCMC)")
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    divider = make_axes_locatable(ax[2, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    im = ax[2, 2].imshow(x_std_mcmc, cmap=cmap)
    ax[2, 2].set_title("Standard Deviation (MCMC)")
    ax[2, 2].set_xticks([])
    ax[2, 2].set_yticks([])
    divider = make_axes_locatable(ax[2, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    # remove axes[1,0] and axes[2,0]
    ax[1, 0].axis("off")
    ax[2, 0].axis("off")


    #plt.tight_layout()
    plt.show()

# ==============================================================================
# plot
post_stats_plots(x_true, y_true, y_meas, x_mean_bbvi, x_std_bbvi, x_mean_mcmc, x_std_mcmc)