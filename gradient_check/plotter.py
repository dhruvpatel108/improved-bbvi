# import libraries
import os
import numpy as np
import config
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
fontsize_labels = 16
linewidth_ = 1.5
markersize_ = 16
# =============================================================================

x_dim = [2, 10, 50]
#xi_sampling = ["normal_symmetric", "normal_symmetric"]
#xi_sampling_names = ["Symmetric Normal", "Symmetric Normal"]
xi_sampling = ["normal", "normal_symmetric", "nonuniform_deterministic"]
xi_sampling_names = ["Normal", "Symmetric Normal", "Non-Uniform Deterministic"]

batch_size_list = [10, 100, 1000, 10000, 100000] #[2, 10, 50, 100, 1000]
bs_list_uniform = [100, 1000, 10000, 100000]

sigma_init_list = [0.70710678]
mu_init_list = [0.]#, 2.5, 4.9]#, 5.1, 7.5, 10.]
color = ["blue"]#, "orange", "green"]#, "red", "purple", "brown"]


for ss, sigma_init in enumerate(sigma_init_list):
    fig, ax = plt.subplots(len(xi_sampling), len(x_dim), figsize=(4.9*(len(x_dim)+1), 5.5*len(xi_sampling)))
    ax1 = plt.gca()
    color_ = []
    for kk in range(len(mu_init_list)):
        color_.append(next(ax1._get_lines.prop_cycler)['color'])

    for jj, x_dim_ in enumerate(x_dim):
        for ii, xi_sampling_ in enumerate(xi_sampling):
            # load data
            if xi_sampling_ == "nonuniform_deterministic":
                data_dir = f"./exps/xdim{x_dim_}_xi{xi_sampling_}/sigma{sigma_init}"
            else:
                data_dir = f"./exps/xdim{x_dim_}_xi{xi_sampling_}"
            error_array = np.load(f"{data_dir}/error_bbvi.npy")
            error_cv_array = np.load(f"{data_dir}/error_bbvi_cv.npy")
            error_fd_array = np.load(f"{data_dir}/error_bbvi_fd.npy")
            error_fd_cv_array = np.load(f"{data_dir}/error_bbvi_fd_cv.npy")
            #mu_error_array = np.load(f"./exps/error_bbvi_xdim{x_dim_}_xi{xi_sampling_}.npy")
            #mu_error_cv_array = np.load(f"./exps/error_bbvi_cv_xdim{x_dim_}_xi{xi_sampling_}.npy")
            #print(mu_error_array.shape, mu_error_cv_array.shape)
            #batch_size_list = np.load(f"batch_size_list_{x_dim_}_{xi_sampling_}.npy")
            #mu_init_list = np.load(f"mu_init_list_{x_dim_}_{xi_sampling_}.npy")
            print(f"x_dim = {x_dim_}, xi_sampling = {xi_sampling_}, \
                  error_array.shape = {error_array.shape}, error_cv_array.shape = {error_cv_array.shape}, \
                  error_fd_array.shape = {error_fd_array.shape}, error_fd_cv_array.shape = {error_fd_cv_array.shape}")
            
            # plot
            for mm, mu_init in enumerate(mu_init_list):
                if (xi_sampling_ == "nonuniform_deterministic") and (x_dim_ >= 10):
                    ax[ii, jj].loglog(bs_list_uniform, error_array[ss, mm, :, 0], "o-", color=color_[mm], 
                            markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI")
                    ax[ii, jj].loglog(bs_list_uniform, error_fd_array[ss, mm, :, 0], "s-", color=color_[mm],
                            markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI + FD")
                    ax[ii, jj].loglog(bs_list_uniform, error_cv_array[ss, mm, :, 0], "x-", color=color_[mm],
                            markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI + CV")
                    ax[ii, jj].loglog(bs_list_uniform, error_fd_cv_array[ss, mm, :, 0], "d-", color=color_[mm],
                            markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI + FD + CV")
                else:
                    ax[ii, jj].loglog(batch_size_list, error_array[ss, mm, :, 0], "o-", color=color_[mm], 
                                markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI")
                    ax[ii, jj].loglog(batch_size_list, error_fd_array[ss, mm, :, 0], "s-", color=color_[mm],
                                markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI + FD")
                    ax[ii, jj].loglog(batch_size_list, error_cv_array[ss, mm, :, 0], "x-", color=color_[mm], 
                                markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI + CV")
                    ax[ii, jj].loglog(batch_size_list, error_fd_cv_array[ss, mm, :, 0], "d-", color=color_[mm],
                                markerfacecolor='none', linewidth=linewidth_, markersize=markersize_, label=f"BBVI + FD + CV")
                    
            ax[ii, jj].grid(True)
            #ax[ii, jj].set_title(rf"$x_{{dim}}$ = {x_dim_}", 
            #ax[ii, jj].set_title(rf"$x_dim$ = {x_dim_}", 
            #                     "\n"
            #                     rf"$\xi$-sampling = {xi_sampling_names[ii]}", fontsize=fontsize_labels)
            ax[ii, jj].set_title(rf"$x_{{dim}}$ = {x_dim_}" + "\n" + rf"$\xi$-sampling = {xi_sampling_names[ii]}", fontsize=fontsize_labels)
            #ax[ii, jj].set_title(rf"$x_{{dim}}$ = {x_dim_}  $\xi$-sampling={xi_sampling_names[ii]}", fontsize=fontsize_labels)
            if ii == len(xi_sampling)-1:
                ax[ii, jj].set_xlabel("Batch size", fontsize=fontsize_labels)
            ax[ii, jj].set_ylabel("Relative error (%)", fontsize=fontsize_labels)
            # only add legend if it is last column and add it to the outside on the right side of last subplot
            if (jj == len(x_dim)-1) and ii==0:
                ax[ii, jj].legend(fontsize=fontsize_labels, bbox_to_anchor=(1.05, 1), loc='upper left')


    fig.tight_layout()
    fig.savefig(f"./exps/error_bbvi_sigma{sigma_init}_SinglePoint.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"./exps/error_bbvi_sigma{sigma_init}_SinglePoint.pdf")
plt.show()
        
