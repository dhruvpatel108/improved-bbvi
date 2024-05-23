# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 21:55:22 2022

@author: Harikrishna
"""
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(0)
plt.close("all")

n_samples = 10000
Nz, Nx = 40, 80 
ktrue = np.load("true-1610x756.npy")
kappa = np.array(Image.fromarray(ktrue).resize((Nx, Nz), Image.NEAREST))
kappa_ud = np.flipud(kappa)


f85 = 0.0199
f75 = 0.0161
f4030 = 0.0642
background_sand = 0.26

# size of each block [size_z x size_x]
box_size = np.zeros([8, 2]).astype(np.int32)
box_size[0, 0] = 4
box_size[0, 1] = 22
box_size[1, 0] = 4
box_size[1, 1] = 18
box_size[2, 0] = 4
box_size[2, 1] = 19
box_size[3, 0] = 4
box_size[3, 1] = 15
box_size[4, 0] = 4
box_size[4, 1] = 16
box_size[5, 0] = 4 
box_size[5, 1] = 15
box_size[6, 0] = 5
box_size[6, 1] = 29
box_size[7, 0] = 5
box_size[7, 1] = 29


#x coord
x1 = np.random.randint(low=6, high=52, size=n_samples)
x2 = np.random.randint(low=6, high=21, size=n_samples)
x3 = np.random.randint(low=40, high=55, size=n_samples)
x4 = np.random.randint(low=4, high=11, size=n_samples)
x5 = np.random.randint(low=30, high=38, size=n_samples)
x6 = np.random.randint(low=57, high=62, size=n_samples)
x7 = np.random.randint(low=4, high=10, size=n_samples)
x8 = np.random.randint(low=41, high=47, size=n_samples)

# y coord
y1 = np.random.randint(low=6, high=9, size=n_samples)
y2 = np.random.randint(low=15, high=18, size=n_samples)
y3 = np.random.randint(low=24, high=27, size=n_samples)
y4 = np.random.randint(low=32, high=35, size=n_samples)

data = np.ones([n_samples, Nz, Nx])*background_sand
for n in range(n_samples):
    # box 1 (In top row i.e. use y1)
    data[n, y1[n]:y1[n]+box_size[0,0], x1[n]:x1[n]+box_size[0,1]] = f85
    
    # box 2 (In second row i.e. use y2)
    data[n, y2[n]:y2[n]+box_size[1,0], x2[n]:x2[n]+box_size[1,1]] = f75
    # box 3 (In second row i.e. use y2)
    data[n, y2[n]:y2[n]+box_size[2,0], x3[n]:x3[n]+box_size[2,1]] = f85
    
    # box 4 (In third row i.e. use y3)
    data[n, y3[n]:y3[n]+box_size[3,0], x4[n]:x4[n]+box_size[3,1]] = f85
    # box 5 (In third row i.e. use y3)
    data[n, y3[n]:y3[n]+box_size[4,0], x5[n]:x5[n]+box_size[4,1]] = f75
    # box 6 (In third row i.e. use y3)
    data[n, y3[n]:y3[n]+box_size[5,0], x6[n]:x6[n]+box_size[5,1]] = f75
    
    # box 7 (In fourth row i.e. use y4)
    data[n, y4[n]:y4[n]+box_size[6,0], x7[n]:x7[n]+box_size[6,1]] = f4030
    # box 8 (In fourth row i.e. use y4)
    data[n, y4[n]:y4[n]+box_size[7,0], x8[n]:x8[n]+box_size[7,1]] = f85


# Normalization
min_value = f75
max_value = background_sand
data_normalized = ((data-min_value)/(max_value - min_value))*2 - 1

for n in range(20):
    plt.figure()
    plt.subplot(121)
    plt.imshow(data[n, :, :])
    plt.title(f"orignal k values = {n}")
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(data_normalized[n, :, :])
    plt.title(f"normalized k values = {n}")
    plt.colorbar()
    
np.save(f"./xy_location_variation_data{n_samples}.npy", data_normalized)    