import numpy as np
import matplotlib.pyplot as plt
import scipy

# ====================================================================================================
# function to compute KL divergence between two distributions
def kl_divergence(p, q):
    return np.sum(p*np.log(p/q))

# ====================================================================================================
# function to compute Maximum Mean Discrepancy (MMD) with Gaussian kernel between two distributions
def mmd(x_true, x_fake):
    """
    computes MMD between two distributions using samples from these distributions (using Gaussian kernel).
    reference1: https://arxiv.org/pdf/1505.03906.pdf
    reference2: https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    reference3: https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
    x_true(NxD): np.array
    x_fake(NxD): np.array
    """
    x_true_dist = scipy.spatial.distance.pdist(x_true)     
    x_fake_dist = scipy.spatial.distance.pdist(x_fake)     
    x_true_fake = scipy.spatial.distance.cdist(x_true, x_fake)

    mmd_true_b = (2*np.sum(np.exp(-0.5*(x_true_dist**2))) + (x_true.shape[0]*np.exp(-0.5)))/(x_true.shape[0]**2)
    mmd_fake_b = (2*np.sum(np.exp(-0.5*(x_fake_dist**2))) + (x_fake.shape[0]*np.exp(-0.5)))/(x_fake.shape[0]**2)
    mmd_cross_b = np.sum(np.exp(-0.5*(x_true_fake**2)))/(x_fake.shape[0]*x_true.shape[0])

    mmd_true_u = (np.sum(np.exp(-0.5*(x_true_dist**2))))/(x_true.shape[0]*(x_true.shape[0]-1)) 
    mmd_fake_u = (np.sum(np.exp(-0.5*(x_fake_dist**2))))/(x_fake.shape[0]*(x_fake.shape[0]-1))
    mmd_cross_u = mmd_cross_b


    return mmd_true_b + mmd_fake_b - (2*mmd_cross_b), mmd_true_u + mmd_fake_u - (2*mmd_cross_u)



def data_processing(x_dim, batch_size, prior_mean_vec, prior_var_vec, like_var_vec, y_meas):
    # prior
    batched_prior_mean_vec = np.repeat(prior_mean_vec.T, batch_size, axis=0)      
    batched_prior_var_vec = np.repeat(prior_var_vec.T, batch_size, axis=0)  
    batched_like_var_vec = np.repeat(like_var_vec.T, batch_size, axis=0)   
    # like
    batched_y_meas = np.repeat(y_meas.T, batch_size, axis=0)     
    # variational
    #mu = np.ones([x_dim, 1])*mu_init
    #mu_2d = np.repeat(mu.T, batch_size, axis=0)  
    #sigma = np.sqrt(0.5)       
    assert batch_size%2 == 0, "batch_size must be even!"
    xi_orig = np.random.multivariate_normal(mean=np.zeros(x_dim), cov=np.eye(x_dim), size=int(batch_size/2))
    xi_neg = -xi_orig
    xi = np.concatenate((xi_orig, xi_neg), axis=0)
    print(f"xi.shape: {xi.shape}")
    #xi = np.linspace(-4, 4, batch_size).reshape([batch_size, 1]])
    #xi = np.random.multivariate_normal(mean=np.zeros(x_dim), cov=np.eye(x_dim), size=batch_size)
    #xx = xi*sigma + mu_2d

    return batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi


def ELBO(xi, mu_list, sigma_list, n_iter, batch_size, x_dim, prior, likelihood):
    assert n_iter==mu_list.shape[0]
    elbo_list = []
    for ii in range(n_iter):
        mu = np.expand_dims(mu_list[ii, :].T, axis=1)
        sigma = np.expand_dims(sigma_list[ii, :].T, axis=1)
        batched_mu = np.repeat(mu.T, batch_size, axis=0)
        batched_sigma = np.repeat(sigma.T, batch_size, axis=0)
        xx = xi*batched_sigma + batched_mu
        L_batch = 0.
        for n in range(batch_size):
            xn = np.expand_dims(xx[n, :], axis=1)
            # prior
            log_prior = prior.log_prior_sample(xn)
            #log_prior = -((x_dim/2)*np.log(2*np.pi)) - (0.5*(np.linalg.norm(xn - prior_mean_vec)**2))
            # likelihood
            log_like = likelihood.log_likelihood_sample(xn)
            #diff1d = y_meas - (A@xn)
            #log_like = -((x_dim/2)*np.log(2*np.pi)) - (0.5*(np.linalg.norm(diff1d)**2))
            # variational 
            log_var = -(0.5*x_dim*np.log(2*np.pi)) - np.sum(np.log(sigma)) - (0.5*np.linalg.norm((xn - mu)/sigma)**2)
            #log_var = -((x_dim/2)*np.log(2*np.pi)) - (x_dim*np.log(sigma)) - ((np.linalg.norm(xn - mu)**2)/(2*(sigma**2)))
            # elbo contribution
            Ln = log_like + log_prior - log_var
            L_batch = L_batch + Ln
        elbo_list.append(L_batch/batch_size)
    return elbo_list




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
    return (np.linalg.norm(true-pred)/(np.linalg.norm(true) + 1e-12))*100

def rel_l1_error(true, pred):
    assert true.shape == pred.shape, f"true and pred must have the same shape! But it is true.shape = {true.shape} and pred.shape = {pred.shape}"
    return (np.linalg.norm(true-pred, ord=1)/(np.linalg.norm(true, ord=1))+1e-12)*100

def rel_scaled_l2_error(true, pred):
    assert true.shape == pred.shape, f"true and pred must have the same shape!"
    return np.linalg.norm((true - pred)/(true+1e-12))*100

def rel_scaled_l1_error(true, pred):
    assert true.shape == pred.shape, f"true and pred must have the same shape!"
    return np.linalg.norm((true - pred)/(true+1e-12), ord=1)*100