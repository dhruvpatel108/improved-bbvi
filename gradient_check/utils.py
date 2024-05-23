import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config
import scipy.stats as stats
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def data_processing(xi_sampling, z_dim, batch_size, prior_mean_vec, prior_var_vec, like_var_vec, y_meas):
    # prior
    batched_prior_mean_vec = np.repeat(prior_mean_vec.T, batch_size, axis=0)      
    batched_prior_var_vec = np.repeat(prior_var_vec.T, batch_size, axis=0)  
    # likelihood
    batched_like_var_vec = np.repeat(like_var_vec.T, batch_size, axis=0)   
    batched_y_meas = np.repeat(y_meas.T, batch_size, axis=0)     
    
    # xi      
    if batch_size==1:
        xi = np.random.multivariate_normal(mean=np.zeros(z_dim), cov=np.eye(z_dim), size=1)
    else:
        if xi_sampling=="normal_symmetric":
            assert batch_size%2 == 0, "batch_size must be even!"
            xi_orig = np.random.multivariate_normal(mean=np.zeros(z_dim), cov=np.eye(z_dim), 
                                                    size=int(batch_size/2))
            xi_neg = -xi_orig
            xi = np.concatenate((xi_orig, xi_neg), axis=0)

        elif xi_sampling=="normal":
            xi = np.random.multivariate_normal(mean=np.zeros(z_dim), cov=np.eye(z_dim), size=batch_size)

        elif xi_sampling=="nonuniform_deterministic":            
            assert batch_size%z_dim == 0, "batch_size must be a multiple of z_dim!"
            n_pts_per_dim = int(batch_size/z_dim)
            xi = np.zeros((batch_size, z_dim))
            start_pt = stats.norm.ppf(1/n_pts_per_dim, loc=0, scale=1)
            for ii in range(z_dim):
                xi[ii*n_pts_per_dim, ii] = start_pt
                for jj in range(1, n_pts_per_dim):
                    xi[ii*n_pts_per_dim+jj, ii] = xi[(ii*n_pts_per_dim)+jj-1, ii] + 1/(n_pts_per_dim*stats.norm.pdf(xi[(ii*n_pts_per_dim)+jj-1, ii], loc=0, scale=1))

        elif xi_sampling=="uniform_deterministic":
            assert batch_size%z_dim == 0, "batch_size must be a multiple of z_dim!"
            n_pts_per_dim = int(batch_size/z_dim)
            xi = np.zeros((batch_size, z_dim))
            for ii in range(z_dim):
                xi[ii*n_pts_per_dim:(ii+1)*n_pts_per_dim, ii] = np.linspace(-1, 1, n_pts_per_dim)
                
        else:
            raise ValueError("xi_sampling must be either \
                             normal_symmetric, normal, nonuniform_deterministic, or uniform_deterministic!")

    return batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi

def log_batched_var_gauss(z, mu, sigma):
    # z: (batch_size, z_dim)
    # mu: (batch_size, z_dim)
    # sigma: (batch_size, z_dim)
    assert z.shape[0]==mu.shape[0] and z.shape[0]==sigma.shape[0], "z, mu, and sigma must have the same batch_size!"
    log_var = -(0.5*z.shape[1]*np.log(2*np.pi)) - np.sum(np.log(sigma), axis=1) - (0.5*np.sum(((z - mu)/sigma)**2, axis=1))
    return log_var


def ELBO_batch(mu_vec, sigma_vec, prior, likelihood, batch_size):
    # sample from the variational distribution
    z_var = np.random.multivariate_normal(mean=mu_vec.squeeze(), cov=np.diag(sigma_vec.squeeze()**2), size=batch_size)
    # prior
    log_prior = prior.log_prior_batch_allterms(z_var)   # (batch_size, )
    # likelihood
    log_like = likelihood.log_likelihood_batch_allterms(z_var)   # (batch_size, )
    # variational
    mu_vec_batched = np.repeat(mu_vec.T, batch_size, axis=0)
    sigma_vec_batched = np.repeat(sigma_vec.T, batch_size, axis=0)
    log_var = log_batched_var_gauss(z_var, mu_vec_batched, sigma_vec_batched)   # (batch_size, )
    # ELBO
    elbo = np.mean(log_like + log_prior - log_var)
    return elbo





# function to plot the posterior statistics
def post_stats_plots(x_true, x_map, x_mean, x_std, save_dir, file_postfix, if_map):
    cmap = "viridis"
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    im = ax[0, 0].imshow(x_true, cmap=cmap)
    ax[0, 0].set_title("True")
    #ax[0, 0].set_xticks([])
    #ax[0, 0].set_yticks([])
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")    
    if if_map:
        im = ax[0, 1].imshow(x_map, cmap=cmap, vmin=np.min(x_true), vmax=np.max(x_true))
        ax[0, 1].set_title("MAP")
        #ax[0, 1].set_xticks([])
        #ax[0, 1].set_yticks([])
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
    im = ax[1, 0].imshow(x_mean, cmap=cmap, vmin=np.min(x_true), vmax=np.max(x_true))
    ax[1, 0].set_title("Mean")
    #ax[1, 0].set_xticks([])
    #ax[1, 0].set_yticks([])
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    im = ax[1, 1].imshow(x_std, cmap=cmap, vmin=0, vmax=np.max(x_true))
    ax[1, 1].set_title("Standard Deviation")
    #ax[1, 1].set_xticks([])
    #ax[1, 2].set_yticks([])
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")  
    #ax[1, 0].axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{file_postfix}.png")



# function to plot the evolution of variational parameters
def variational_params_plots(elbo_list, mu_list, sigma_list, save_dir):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].plot(elbo_list)
    ax[0].set_title("Evolution of ELBO")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("ELBO")
    ax[0].grid()
    ax[1].plot(mu_list)
    ax[1].set_title("Evolution of $\mu$")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("$\mu$")
    ax[1].grid()
    ax[2].plot(sigma_list)
    ax[2].set_title("Evolution of $\sigma$")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("$\sigma$")
    ax[2].grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/var_params.png")
    #plt.show()


def plots(save_dir, mu_list, sigma_list, elbo_list, true_mu_post, mu_opt, true_std_post, sigma_opt, noise_var, mu_step_size, gamma_step_size, 
            batch_size, n_iter, x_dim, true_post_samples, mu_l2_error, sigma_l2_error, mu_l1_error, sigma_l1_error,  
            mu_scaled_l2_error, sigma_scaled_l2_error, mu_scaled_l1_error, sigma_scaled_l1_error):
    # find the indices of the mu and sigma values that are farthest to the true posterior mean and std in l2 sense
    mu_diff = np.abs(mu_opt - true_mu_post)
    sigma_diff = np.abs(sigma_opt - true_std_post)
    mu_index = np.argmax(mu_diff)
    sigma_index = np.argmax(sigma_diff)
    print(f"mu_index: {mu_index}, sigma_index: {sigma_index} <<<<<<<<<<<<<")
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(elbo_list)
    plt.grid(True)
    plt.xlabel("iteration")
    plt.ylabel("ELBO")    
    plt.subplot(132)
    plt.plot(true_mu_post[mu_index]*np.ones(n_iter), label="true")
    plt.plot(mu_list[:, mu_index], label="estimated")
    plt.grid(True)
    plt.xlabel("iteration")
    plt.ylabel(f"$\mu_{{{mu_index}}}$")
    plt.subplot(133)
    plt.plot(true_std_post[sigma_index]*np.ones(n_iter), label="true")
    plt.plot(sigma_list[:, sigma_index], label="estimated")
    plt.grid(True)
    plt.xlabel("iteration")
    plt.ylabel(f"$\sigma_{{{sigma_index}}}$")
    plt.suptitle(f"noise_var = {noise_var} | mu_step_size = {mu_step_size} | gamma_step_size = {gamma_step_size} | batch_size = {batch_size} | n_iter = {n_iter} | \n \
                    mu_l2_error = {mu_l2_error:.2f} % | sigma_l2_error = {sigma_l2_error:.2f} % | mu_l1_error = {mu_l1_error:.2f} % | sigma_l1_error = {sigma_l1_error:.2f} % \n \
                    mu_scaled_l2_error = {mu_scaled_l2_error:.2f} | sigma_scaled_l2_error = {sigma_scaled_l2_error:.2f} | mu_scaled_l1_error = {mu_scaled_l1_error:.2f} | sigma_scaled_l1_error = {sigma_scaled_l1_error:.2f}", fontsize=12)    
    # var_post_samples
    var_post_samples = np.random.multivariate_normal(mean=mu_opt.squeeze(), cov=np.eye(x_dim) * (sigma_opt**2), size=1000)    
    plt.figure(figsize=(15, 15))
    plt.plot(true_post_samples[:, 0], true_post_samples[:, 1], "b o", alpha=0.2, label="true_samples")
    plt.plot(var_post_samples[:, 0], var_post_samples[:, 1], "r o", alpha=0.2, label="var_samples")
    plt.legend()
    plt.savefig(f"{save_dir}/log_plot")

    # 

    # find the k indices for which mu_diff is the largest
    k = min(16, x_dim)
    mu_highest_idx = np.argpartition(mu_diff, -k)[-k:][::-1]
    sigma_highest_idx = np.argpartition(sigma_diff, -k)[-k:][::-1]
    print(f"mu_highest_idx: {mu_highest_idx}, sigma_highest_idx: {sigma_highest_idx} <<<<<<<<<<<<<")
    print(f"mu_diff: {mu_diff[mu_highest_idx]} <<<<<<<<<<<<<")
    print(f"sigma_diff: {sigma_diff[sigma_highest_idx]} <<<<<<<<<<<<<")

    #plot the true mean vector and the estimated mean vector at each iteration for the k indices
    plt.figure(figsize=(15, 15))
    for ii, idx in enumerate(mu_highest_idx):
        #print(f"plotting mu_{idx}")
        plt.subplot(4, 4, ii+1)
        plt.plot(mu_list[:, idx], label="estimated")
        plt.plot(true_mu_post[idx]*np.ones(n_iter), label="true")
        plt.ylabel(fr"$\mu_{{{idx}}}$")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/highest_mu_diff")

    #plot the true std vector and the estimated std vector at each iteration for the k indices
    plt.figure(figsize=(15, 15))
    for ii, idx in enumerate(sigma_highest_idx):
        #print(f"plotting sigma_{idx}")
        plt.subplot(4, 4, ii+1)
        plt.plot(sigma_list[:, idx], label="estimated")
        plt.plot(true_std_post[idx]*np.ones(n_iter), label="true")
        plt.ylabel(fr"$\sigma_{{{idx}}}$")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/highest_sigma_diff")




    # plot the true mean vector and the estimated mean vector at each iteration
    n_plots = min(16, x_dim)
    assert n_plots > 0, f"n_plots must be greater than 0! But it is {n_plots}"
    idx_array = np.linspace(0, x_dim-1, n_plots).astype(np.int)
    print(f"idx_array: {idx_array} <<<<<<<<<<<<<")
    plt.figure(figsize=(15, 15))
    for ii, idx in enumerate(idx_array):
        #print(f"plotting mu_{idx}")
        plt.subplot(int(np.sqrt(n_plots)), int(np.sqrt(n_plots)), ii+1)
        plt.plot(mu_list[:, idx], label="estimated")
        plt.plot(true_mu_post[idx]*np.ones(n_iter), label="true")
        plt.grid(True)
        plt.xlabel("iteration")
        plt.ylabel(f"$\mu_{{{idx}}}$")
        plt.legend()
        plt.savefig(f"{save_dir}/mu_plot")

    plt.figure(figsize=(15, 15))
    for ii, idx in enumerate(idx_array):
        #print(f"plotting sigma_{idx}")
        plt.subplot(int(np.sqrt(n_plots)), int(np.sqrt(n_plots)), ii+1)
        plt.plot(sigma_list[:, idx], label="estimated")
        plt.plot(true_std_post[idx]*np.ones(n_iter), label="true")
        plt.grid(True)
        plt.xlabel("iteration")
        plt.ylabel(f"$\sigma_{{{idx}}}$")
        plt.legend()
        plt.savefig(f"{save_dir}/sigma_plot")     


    plt.show()

def rel_l2_error(true, pred):
    assert true.shape == pred.shape, f"true and pred must have the same shape! But it is true.shape = {true.shape} and pred.shape = {pred.shape}"
    return (np.linalg.norm(true-pred)/np.linalg.norm(true))*100

def rel_l1_error(true, pred):
    assert true.shape == pred.shape, f"true and pred must have the same shape! But it is true.shape = {true.shape} and pred.shape = {pred.shape}"
    return (np.linalg.norm(true-pred, ord=1)/np.linalg.norm(true, ord=1))*100

def rel_scaled_l2_error(true, pred):
    assert true.shape == pred.shape, f"true and pred must have the same shape!"
    return np.linalg.norm((true - pred)/(true+1e-12))*100

def rel_scaled_l1_error(true, pred):
    assert true.shape == pred.shape, f"true and pred must have the same shape!"
    return np.linalg.norm((true - pred)/(true+1e-12), ord=1)*100