import numpy as np
import matplotlib.pyplot as plt
import config
import scipy
from utils import *
from models import *
from optimizers import *
plt.rcParams['font.size'] = 14/4
np.random.seed(1008)
plt.close("close all")
# parameters
like_var = 1.0
prior_var = 1.0
y_meas = config.y_meas
prior_mean_vec = config.prior_mean_vec.squeeze()
prior_var_vec = config.prior_var_vec.squeeze()
print(prior_mean_vec.shape)
print(prior_var_vec)
#A = 1.0


# Define the forward model
def forward_model(x):
    return 0.2*(x-2)**3*np.sin(x-2) #+ 0.2*(x-2)**2*np.sin(x-2)

def dfdx(x):
    return 0.6*(x-2)**2*np.sin(x-2) + 0.2*(x-2)**3*np.cos(x-2) #+ 0.4*(x-2)*np.sin(x-2) + 0.2*(x-2)**2*np.cos(x-2)

def hessian(x):
    return 1.2*(x-2.)*np.sin(x-2.) + 0.6*(x-2.)**2.*np.cos(x-2.) + 0.6*(x-2.)**2*np.cos(x-2.) - 0.2*(x-2.)**3*np.sin(x-2.)


# Define the log-likelihood function
log_like = lambda x: -0.5 * np.log(2 * np.pi * like_var) - 0.5 * ((forward_model(x) - y_meas.squeeze())**2 / like_var)
# Define the log-prior function
#log_prior = lambda x: -0.5 * np.log(2 * np.pi * prior_var) - 0.5 * (x**2 / prior_var)
log_prior = lambda x: -0.5 * np.log(2 * np.pi * prior_var_vec) - 0.5 * ((x-prior_mean_vec)**2 / prior_var_vec)
# Define the log-joint function
log_joint = lambda x: log_like(x) + log_prior(x)
# plot the Gaussian approximation using mu_opt and sigma_opt
log_gaussian = lambda z, mu, sigma: -0.5*np.log(2*np.pi*(sigma**2)) - 0.5*((z-mu)**2/sigma**2)


##
xl = 0.0 # x_prior 
for i in range(100):
    dx = -(1./prior_var_vec*(xl - 2.0) + dfdx(xl)*(forward_model(xl)-y_meas.squeeze())/like_var)/(1./prior_var + dfdx(xl)**2./like_var)
    xl = xl + dx 
    print('iter %d: log_joint = %.4f, x = %.4f, dx = %.4f' % (i, log_joint(xl), xl, dx))
    if np.abs(dx) < 1e-6:
        break

post_var = 1./(1./prior_var + dfdx(xl)**2./like_var + hessian(xl)*(forward_model(xl)-y_meas.squeeze())/like_var)

#post_var = prior_var - prior_var**2*dfdx(xl)**2/(prior_var + like_var*dfdx(xl)**2)
##


# find the mean and variance of the log-joint function
def find_mean_var(log_prob, z_min, z_max, *args):
    z = np.linspace(z_min, z_max, 1000000)
    y = np.exp(log_prob(z, *args))
    mean = np.sum(z*y)/np.sum(y)
    var = np.sum((z-mean)**2*y)/np.sum(y)
    return mean, var


n_samples = 1000
# function to compute the normalization constant of the log_prob function
def compute_normalization(log_prob, z_min, z_max, *args):
    z = np.linspace(z_min, z_max, n_samples)
    y = np.exp(log_prob(z, *args))
    return np.sum(y)*(z_max-z_min)/n_samples
normalization_log_joint = compute_normalization(log_joint, -9, 9)
#normalization_log_gaussian = compute_normalization(log_gaussian, -9, 9, 0, np.sqrt(0.5))

# posterior distribution
def posterior_dist(z):
    return np.exp(log_joint(z))/normalization_log_joint



# Plot the log-likelihood, log-prior and log-joint functions
z = np.linspace(-5+2, 5+2, 128)
plt.plot(z, log_like(z), label="log-likelihood")
plt.plot(z, log_prior(z), label="log-prior")
plt.plot(z, log_joint(z), label="log-joint")
plt.ylim([-7, 5])
plt.legend()
plt.grid()


def quadratic_approx(z1, y1, z2, y2, z3, y3):
    a = ((y3-y1)*(z2-z1) - (y2-y1)*(z3-z1))/((z3-z1)*(z2-z1)*(z3-z2))
    b = ((y2-y1)/(z2-z1) - a*(z2+z1))
    c = y1 - a*z1**2 - b*z1
    return a, b, c

# Compute the quadratic approximation of the log-joint function
z2 = config.mu_init_scalar
sigma = 0.05
z1 = z2-sigma
z3 = z2+sigma
y1 = log_joint(z1)
y2 = log_joint(z2)
y3 = log_joint(z3)
a, b, c = quadratic_approx(z1, y1, z2, y2, z3, y3)
quad_it = 0
var_quad = -1/(2*a)
mean_quad = -b/(2*a)

while np.abs(mean_quad-z2) > 1e-3:
    z2 = mean_quad
    z1 = z2-sigma
    z3 = z2+sigma
    y1 = log_joint(z1)
    y2 = log_joint(z2)
    y3 = log_joint(z3)
    a, b, c = quadratic_approx(z1, y1, z2, y2, z3, y3)
    var_quad = -1/(2*a)
    mean_quad = -b/(2*a)
    #sigma = np.sqrt(var_quad)
    quad_it += 1
    print(f"quad_it = {quad_it} mean_quad = {mean_quad:.4f}, var_quad = {var_quad:.4f}")


print(f"(z1, y1) = ({z1:.4f}, {y1:.4f}), (z2, y2) = ({z2:.4f}, {y2:.4f}), (z3, y3) = ({z3:.4f}, {y3:.4f})")
print(f"z2 = {z2:.4f}, sigma = {sigma:.4f} | a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
print(f"no. of quad iterations = {quad_it}")



# ====================================================================================================
# BBVI approximation of the posterior distribution
# ====================================================================================================
xi_orig_1d = np.random.multivariate_normal(mean=np.zeros(config.x_dim), cov=np.eye(config.x_dim), size=1)    
batched_prior_mean_vec, batched_prior_var_vec, batched_like_var_vec, batched_y_meas, xi = data_processing(config.x_dim, config.batch_size, config.prior_mean_vec, config.prior_var_vec, config.like_var_vec, config.y_meas)
prior = Prior(config.prior_mean_vec, config.prior_var_vec, batched_prior_mean_vec, batched_prior_var_vec)
likelihood = Likelihood(config.y_meas, batched_y_meas, forward_model, config.like_var_vec, batched_like_var_vec)
if config.optimizer == "gradient_descent":
    mu_opt, gamma_opt, mu_list_np, gamma_list_np = gradient_descent(xi, xi_orig_1d, config.mu_init_guess, config.gamma_init_guess, 
                                                                    config.n_iter, config.mu_step_size, config.gamma_step_size, prior, likelihood, 
                                                                    if_fd_correction=True, if_cv=True)  

elif config.optimizer == "adam":
    mu_opt, gamma_opt, mu_list_np, gamma_list_np = adam(xi, xi_orig_1d, config.mu_init_guess, config.gamma_init_guess, 
                                                        config.n_iter, config.mu_step_size, config.gamma_step_size, prior, likelihood, 
                                                        if_fd_correction=True, if_cv=True)

else:
    raise ValueError("Optimizer not supported")

sigma_opt = np.exp(gamma_opt)
sigma_list_np = np.exp(gamma_list_np)
#print(f"mu_opt = {mu_opt.squeeze()} | var_opt = {sigma_opt.squeeze()**2}")

# ====================================================================================================
# MCMC approximation of the posterior distribution
# ====================================================================================================
n_mcmc_samples = config.batch_size*config.n_iter


def random_walk_mcmc(n_mcmc_samples, posterior, st_point):
    z = st_point
    z_list = []
    for i in range(n_mcmc_samples):
        z_prop = z + np.random.normal(0, 0.25, size=z.shape)
        log_ratio = posterior(z_prop) - posterior(z)
        if np.log(np.random.uniform(0, 1)) < log_ratio:
            z = z_prop
        z_list.append(z)
    return np.array(z_list)

z_mcmc = random_walk_mcmc(n_mcmc_samples, log_joint, np.array([config.mu_init_scalar]))
z_mcmc = z_mcmc.reshape([n_mcmc_samples, 1])[int(0.25*n_mcmc_samples):, 0]
mu_mcmc = np.mean(z_mcmc)
sigma_mcmc = np.std(z_mcmc)
print(f"mu_mcmc = {mu_mcmc} | sigma_mcmc = {sigma_mcmc}")



# ====================================================================================================
# plot mu_list and sigma_list
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(mu_list_np, color="b", linewidth=2, label="mu")
plt.ylabel("mu")
plt.subplot(2, 1, 2)
plt.plot(sigma_list_np, color="r", linewidth=2, label="sigma")
plt.ylabel("sigma")

# Plot the log-joint function and the bbvi approximation
plt.figure()
plt.plot(z, log_joint(z), color="b", linewidth=4, label="log-joint")
plt.plot(z, a*z**2 + b*z + c, "-.", color="r", linewidth=2, label="quadratic approximation (log)")
plt.plot(z1, y1, "*", color="r", markersize=10, label="(z1, y1)")
plt.plot(z2, y2, "*", color="r", markersize=10, label="(z2, y2)")
plt.plot(z3, y3, "*", color="r", markersize=10, label="(z3, y3)")
plt.plot(z, log_gaussian(z, mu_opt.squeeze(), sigma_opt.squeeze()).squeeze(), color="g", linewidth=2, label="bbvi approximation (log)")
plt.plot(mu_opt, log_gaussian(mu_opt, mu_opt, sigma_opt), "s", color="g", markersize=10, label="mu_opt")
#plt.plot(z, log_gaussian(z, 0, np.sqrt(0.5)), color="k", linewidth=2, label="N(0, 0.5)")
plt.legend()
plt.grid()
#plt.show()

# ====================================================================================================
var_quad = -1/(2*a)
mean_quad = -b/(2*a)
sigma_quad = np.sqrt(var_quad)
mean_laplace = xl
sigma_laplace = np.sqrt(post_var)
mean_log_joint, var_log_joint = find_mean_var(log_joint, -9, 9)
#mean_log_gaussian, var_log_gaussian = find_mean_var(log_gaussian, -9, 9, mu_opt.squeeze(), sigma_opt.squeeze())
print(f"MC estimate of posterior mean   = {mean_log_joint:.4f} and var = {var_log_joint:.4f}")
#print(f"MC estimate of mean = {mean_log_gaussian:.4f} and var = {var_log_gaussian:.4f} for log_gaussian")
print(f"Mean of bbvi approximation      = {mu_opt.squeeze():.4f} and var = {sigma_opt.squeeze()**2:.4f}")
print(f"Mean of quadratic approximation = {mean_quad:.4f} and var = {var_quad:.4f}")
print(f"Mean of Laplace approximation    = {mean_laplace:.4f} and var = {post_var:.4f}")
print(f"Mean of mcmc approximation      = {mu_mcmc:.4f} and var = {sigma_mcmc:.4f}")
bbvi_mean_pt_error = rel_l2_error(mean_log_joint, mu_opt.squeeze())
bbvi_var_pt_error = rel_l2_error(var_log_joint, sigma_opt.squeeze()**2)
quad_mean_pt_error = rel_l2_error(mean_log_joint, mean_quad)
quad_var_pt_error = rel_l2_error(var_log_joint, var_quad)
laplace_mean_pt_error = rel_l2_error(mean_log_joint, mean_laplace)
laplace_var_pt_error = rel_l2_error(var_log_joint, post_var)
mcmc_mean_pt_error = rel_l2_error(mean_log_joint, mu_mcmc)
mcmc_var_pt_error = rel_l2_error(var_log_joint, sigma_mcmc**2)
print(f"Relative L2 error in mean (BBVI approximation)      = {bbvi_mean_pt_error:.4f}%")
print(f"Relative L2 error in var (BBVI approximation)       = {bbvi_var_pt_error:.4f}%")
print(f"Relative L2 error in mean (quadratic approximation) = {quad_mean_pt_error:.4f}%")
print(f"Relative L2 error in var (quadratic approximation)  = {quad_var_pt_error:.4f}%")
print(f"Relative L2 error in mean (Laplace approximation)   = {laplace_mean_pt_error:.4f}%")
print(f"Relative L2 error in var (Laplace approximation)    = {laplace_var_pt_error:.4f}%")
print(f"Relative L2 error in mean (mcmc approximation)      = {mcmc_mean_pt_error:.4f}%")
print(f"Relative L2 error in var (mcmc approximation)       = {mcmc_var_pt_error:.4f}%")
# compute KL divergence between the posterior distribution and the approximated distributions
kl_bbvi = kl_divergence(np.exp(log_joint(z))/normalization_log_joint, np.exp(log_gaussian(z, mu_opt.squeeze(), sigma_opt.squeeze())).squeeze())
kl_quad = kl_divergence(np.exp(log_joint(z))/normalization_log_joint, np.exp(a*z**2 + b*z + c)/normalization_log_joint)
kl_laplace = kl_divergence(np.exp(log_joint(z))/normalization_log_joint, np.exp(log_gaussian(z, mean_laplace, sigma_laplace)).squeeze())

print(f"KL divergence between posterior and bbvi approximation      = {kl_bbvi:.4f}")
print(f"KL divergence between posterior and quadratic approximation = {kl_quad:.4f}")
print(f"KL divergence between posterior and Laplace approximation    = {kl_laplace:.4f}")





def gaussian_dist(z, mu, sigma):
    return np.exp(-((z-mu)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

# ====================================================================================================

# Plot the probability density function of the posterior distribution and bbvi approximation
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=400)
axes[0].plot(z, posterior_dist(z), color="b", linewidth=4/4, label="posterior")
axes[0].plot(z, gaussian_dist(z, mean_quad, sigma_quad), "-.", color="r", linewidth=2, label="quadratic approximation")
axes[0].plot(z1, gaussian_dist(z1, mean_quad, sigma_quad), "*", color="r", markersize=10, label="(z1, y1)")
axes[0].plot(z2, gaussian_dist(z2, mean_quad, sigma_quad), "*", color="r", markersize=10, label="(mu_quad_opt, y2)")
axes[0].plot(z3, gaussian_dist(z3, mean_quad, sigma_quad), "*", color="r", markersize=10, label="(z3, y3)")
axes[0].legend()
axes[0].grid()
axes[1].plot(z, posterior_dist(z), color="b", linewidth=4/4, label="posterior")
axes[1].plot(z, np.exp(log_gaussian(z, mu_opt.squeeze(), sigma_opt.squeeze())).squeeze(), color="g", linewidth=2, label="bbvi approximation")
axes[1].plot(mu_opt, np.exp(log_gaussian(mu_opt, mu_opt, sigma_opt)), "s", color="g", markersize=10, label="mu_bbvi_opt")
axes[1].legend()
axes[1].grid()
axes[2].plot(z, posterior_dist(z), color="b", linewidth=4/4, label="posterior")
#axes[2].plot(z, np.exp(a*z**2 + b*z + c)/normalization_log_joint, "-.", color="r", linewidth=2, label="quadratic approximation")
#axes[2].plot(z, np.exp(log_gaussian(z, mu_opt.squeeze(), sigma_opt.squeeze())).squeeze(), color="g", linewidth=2, label="bbvi approximation")
axes[2].plot(z, gaussian_dist(z, mean_quad, sigma_quad), "-.", color="r", linewidth=2, label="quadratic approximation")
#axes[2].plot(z1, gaussian_dist(z1, mean_quad, sigma_quad), "*", color="r", markersize=10, label="(z1, y1)")
axes[2].plot(z2, gaussian_dist(z2, mean_quad, sigma_quad), "*", color="r", markersize=10, label="mu_quad_opt")
#axes[2].plot(z3, gaussian_dist(z3, mean_quad, sigma_quad), "*", color="r", markersize=10, label="(z3, y3)")
# bbvi
axes[2].plot(z, gaussian_dist(z, mu_opt.squeeze(), sigma_opt.squeeze()), color="g", linewidth=2, label="bbvi approximation")
axes[2].plot(mu_opt, gaussian_dist(mu_opt, mu_opt.squeeze(), sigma_opt.squeeze()), "s", color="g", markersize=10, label="mu_bbvi_opt")
axes[2].plot(config.mu_init_scalar, 0., "o", color="y", markersize=10, label="mu_init")
axes[2].plot(np.ones(2)*mean_log_joint, [0., 1.], "--", color="b", markersize=10, label="mu_opt")
axes[2].legend()
axes[2].set_ylim([-0.1, 1.1])
axes[2].grid() 



#fig, ax = plt.subplots(ncols=5, dpi=400, gridspec_kw={'width_ratios': [5, 0.1, 3, 3, 2.5]}, figsize=(10, 2), constrained_layout=True)
plt.rcParams['font.size'] = 7
fontsize = 8
fig, ax = plt.subplots(ncols=2, dpi=400, gridspec_kw={'width_ratios': [5, 0.1]}, figsize=(4, 2), constrained_layout=True)
error_type = ['Relative error \n in mean (in %)', 'Relative error \n in var (in %)', 'KL divergence']
labels = ['BBVI', 'MCMC', 'Least-square\napprox.', 'Laplace\napprox.']
colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:cyan']
ax[0].plot(z, posterior_dist(z), color="b", linewidth=4/4, label="True posterior")
ax[0].plot(z, gaussian_dist(z, mean_quad, sigma_quad), "-.", color="r", linewidth=2/4, label="Least-square \napproximation")
ax[0].plot(z, gaussian_dist(z, mu_opt.squeeze(), sigma_opt.squeeze()), color="g", linewidth=2/4, label="BBVI \napproximation")
ax[0].plot(z, gaussian_dist(z, xl, post_var), "--", color="k", linewidth=2/4, label="Laplace \napproximation")

# plot MCMC distribution histogram
ax[0].hist(z_mcmc, bins=75, density=True, alpha=0.25, label="MCMC approximation")

ax[0].plot(config.mu_init_scalar, 0., "o", color="b", markersize=10/4, alpha=0.5, label="Starting point")
#ax[0].plot(np.ones(2)*mean_log_joint, [0., 1.], "-.", color="b", markersize=10/4, label="Mean of the true \nposterior")
#ax[0].plot(z2, gaussian_dist(z2, mean_quad, sigma_quad), "*", color="r", markersize=10/4, label="Mean of the least-square \napproximation")
#ax[0].plot(mu_opt, gaussian_dist(mu_opt, mu_opt.squeeze(), sigma_opt.squeeze()), "s", color="g", markersize=10/4, label="Mean of the BBVI \napproximation")

ax[0].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize=fontsize)
ax[0].set_ylim([-0.1, 1.1])
ax[0].grid() 
ax[0].set_title('$f(x) = 0.2(x-2)^3sin(x-2)$', fontsize=fontsize)

# empty second subplot
ax[1].axis('off')
plt.tight_layout()
plt.savefig('1d_visualization.png', dpi=600)

fig, ax = plt.subplots(ncols=3, dpi=400, gridspec_kw={'width_ratios': [3, 3, 2.5]}, figsize=(6, 2.5), constrained_layout=True)
values_mean = [bbvi_mean_pt_error, mcmc_mean_pt_error, quad_mean_pt_error, laplace_mean_pt_error]
ax[0].bar(x=labels, height=values_mean, color=colors)
ax[0].set_title(error_type[0], fontsize=fontsize)
ax[0].set_xticklabels(labels=labels, rotation=90)
ax[0].set_yscale('log') 
ax[0].set_ylim([1e-1, 2*1e2])
values_var = [bbvi_var_pt_error, mcmc_var_pt_error, quad_var_pt_error, laplace_var_pt_error]
ax[1].bar(x=labels, height=values_var, color=colors)
ax[1].set_title(error_type[1], fontsize=fontsize)
ax[1].set_xticklabels(labels=labels, rotation=90)
ax[1].set_yscale('log') 
ax[1].set_ylim([1e1, 1*1e2])
values_kl = [kl_bbvi, kl_quad, kl_laplace]
ax[2].bar(x=[labels[0], labels[2], labels[3]], height=values_kl, color=[colors[0], colors[2], colors[3]])
ax[2].set_title(error_type[2], fontsize=fontsize)
ax[2].set_xticklabels(labels=[labels[0], labels[2], labels[3]], rotation=90)
ax[2].set_yscale('log') 
ax[2].set_ylim([1e-1, 1*1e2])


# save high resolution figure
#plt.tight_layout()
plt.savefig('1d_errorbar.png', dpi=600)
#plt.savefig('quadratic_approx.pdf')
# ====================================================================================================
# function to generate samples from arbitrary distribution
"""
def sample_from_distribution(distribution, num_samples, z_min, z_max, *args):
    z = np.linspace(z_min, z_max, num_samples)
    y = distribution(z, *args)
    y = y/np.sum(y)
    samples = np.random.choice(z, size=num_samples, replace=True, p=y)
    return samples

# generate samples from the posterior distribution
num_samples = 10000
z_min = -9
z_max = 9
samples_post = sample_from_distribution(posterior_dist, num_samples, z_min, z_max)
samples_bbvi = sample_from_distribution(gaussian_dist, num_samples, z_min, z_max, mu_opt.squeeze(), sigma_opt.squeeze())
samples_quad = sample_from_distribution(gaussian_dist, num_samples, z_min, z_max, mean_quad, sigma_quad)

# plot the histogram of the samples
plt.figure()
plt.hist(samples_post, bins=100, density=True, alpha=0.2, label="posterior")
plt.plot(z, posterior_dist(z), color="b", linewidth=4)
plt.hist(samples_bbvi, bins=100, density=True, alpha=0.2, label="bbvi approximation")
plt.plot(z, gaussian_dist(z, mu_opt.squeeze(), sigma_opt.squeeze()), color="g", linewidth=2)
plt.hist(samples_quad, bins=100, density=True, alpha=0.2, label="quadratic approximation")
plt.plot(z, gaussian_dist(z, mean_quad, sigma_quad), "-.", color="r", linewidth=2)
plt.legend()
 
#mmd_bbvi_b, mmd_bbvi_u = mmd(np.expand_dims(samples_post, axis=1), np.expand_dims(samples_bbvi, axis=1))
#mmd_quad_b, mmd_quad_u = mmd(np.expand_dims(samples_post, axis=1), np.expand_dims(samples_quad, axis=1))
#print(f"MMD between posterior and bbvi approximation (biased)      = {mmd_bbvi_b:.4f} and (unbiased) = {mmd_bbvi_u:.4f}")
#print(f"MMD between posterior and quadratic approximation (biased) = {mmd_quad_b:.4f} and (unbiased) = {mmd_quad_u:.4f}")

"""
plt.show()

