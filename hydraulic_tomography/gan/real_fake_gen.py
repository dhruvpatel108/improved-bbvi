import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
np.random.seed(1008)
from models import Generator
plt.close("all")
# Load generator
ckpt = torch.load("./ckpt/gen_9999 copy.pt", map_location=torch.device("cpu"))
latent_dim = 50
dim = 8
img_size = (40, 80, 1)
min_range = 0.0161
max_range = 0.26
z_sample = np.random.normal(size=[1, latent_dim])
generator = Generator(img_size=img_size, latent_dim=latent_dim, dim=dim)
generator.load_state_dict(ckpt)



def plot_subplots(data):
    n_rows = 6
    n_cols = 6
    fig, axs = plt.subplots(n_rows, n_cols)
    for ii in range(n_rows):
        for jj in range(n_cols):
            im = axs[ii,jj].imshow(data[(ii*n_cols) + jj, :, :])
            axs[ii,jj].axis("off")
            divider = make_axes_locatable(axs[ii,jj])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, ax=axs[ii,jj], cax=cax)

# Real gen
data = np.load("data/xy_location_variation_data10000_fixedf85_f75.npy")
data = ((data+1)/2)*(max_range-min_range) + min_range  
plot_subplots(data)

# Fake gen
fake = []
for n in range(64):
    z = np.squeeze(z_sample)
    z0 = torch.autograd.Variable(torch.tensor(z, dtype=torch.float32), requires_grad=True)
    gz = (((torch.squeeze(generator(z0))+1.)/2)*(max_range-min_range)) + min_range
    print(gz.shape)
    fake.append(gz.detach().cpu().numpy())

print(np.array(fake).shape)
plot_subplots(np.array(fake))
plt.show()
