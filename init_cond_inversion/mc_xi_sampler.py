# Jay Swaminarayan
import numpy as np
import matplotlib.pyplot as plt
import config

seed_no = 1008
N = 12800
a_temp = 0.0
b_temp = 4.0
noise_var = config.noise_var
batch_size = 6400
int_nodes = 30
dim_like = int_nodes**2
like_stddev = np.sqrt(config.like_var)
n_iter = int(N/batch_size)
#y_meas = np.tile(np.expand_dims(np.load('./../forward_model/test127_k0.64_noisevar1.0/u_noisy.npy'), axis=1), (1, batch_size))
#F = np.load('./../forward_model/test127_k0.64_noisevar1.0/F.npy')
np.random.seed(1008)

#print(config.y_meas.shape, config.F.shape)
tl = (np.random.rand(2,N)*0.2)+0.2
br = (np.random.rand(2,N)*0.2)+0.6
tl_ = (tl*28).astype(np.int32)
br_ = (br*28).astype(np.int32)

x_samples = np.zeros((N, 32, 32))
for n in range(N):
    width_vec1 = np.arange(tl_[1,n], br_[1,n])
    o = np.interp(width_vec1, xp=[width_vec1[0], width_vec1[-1]], fp=[0.5, 1])
    x_samples[n, tl_[0,n]:br_[0,n], tl_[1,n]:br_[1,n]] = o*(b_temp-a_temp) + a_temp

    
numerator = np.zeros((batch_size, n_iter))
x_rec = np.zeros((batch_size, 32, 32, n_iter))
for n in range(n_iter):
    x_batch = np.transpose(np.reshape(x_samples[n*batch_size:(n+1)*batch_size, 1:-1, 1:-1], [batch_size, dim_like]))
    print(f"x_batch {x_batch.shape}")
    y_pred = np.matmul(config.F, x_batch)
    diff = y_pred - config.y_meas
    x_rec[:,:,:,n] = x_samples[n*batch_size:(n+1)*batch_size,:,:]
    for k in range(batch_size):
        numerator[k, n] = np.exp(-((np.linalg.norm(diff[:,k])**2)/(2*like_stddev*like_stddev)))

norm_prob = numerator/np.sum(numerator)

x_mean = np.zeros((32, 32)) 
x2_mean = np.zeros((32, 32))
x2_rec = x_rec**2

prob_reshape = np.reshape(norm_prob, [batch_size*n_iter, 1])
xt = np.transpose(x_rec, (1,2,0,3))
xrt = np.reshape(xt, [32, 32, batch_size*n_iter])
xt2 = np.transpose(x2_rec, (1,2,0,3))
xrt2 = np.reshape(xt2, [32, 32, batch_size*n_iter])

x_mean = np.squeeze(np.dot(xrt, prob_reshape))
x2_mean = np.squeeze(np.dot(xrt2, prob_reshape))            
var = x2_mean - (x_mean**2)

descending = np.sort(prob_reshape, axis=0)[::-1]
cummulative = np.cumsum(descending)
effective_samps = int(sum((cummulative<0.99).astype(int)))

print('sum of normailized prob. = {} and shape of it is {}'.format(np.sum(norm_prob), np.shape(norm_prob)))
#print('no. of non-zero norm_probs = {}'.format(np.size(np.nonzero(norm_prob))))
#print('max. norm-prob = {}'.format(np.max(norm_prob)))
#print('no. of effective samples = {}'.format(effective_samps))

#np.save('./test127_k0.64_noisevar1.0_XImean_MC-N{}.npy'.format(N), x_mean)
#np.save('./test127_k0.64_noisevar1.0_XIvar_MC-N{}.npy'.format(N), var)

plt.figure()
plt.subplot(121)
plt.imshow(x_mean)
plt.colorbar()
plt.subplot(122)
plt.imshow(var)
plt.colorbar()
plt.show()
