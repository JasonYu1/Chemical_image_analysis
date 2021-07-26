import matlab.engine
# import numpy as np
import tkinter as tk

# modifiable parameters
sigma = 11
distribution = 'Gauss'
profile = 'mp'
do_wiener = 1
verbose = 1
estimate_sigma = 0
phantom = 't1_icbm_normal_1mm_pn0_rf0.rawb'
crop_phantom = 1
save_mat = 0
variable_noise = 0
noise_factor = 3

###

###
# start matlab engine
eng = matlab.engine.start_matlab()

# read original phantom
y = eng.open_reshape(phantom)
if 'crop_phantom' in locals():
    y = eng.cropdata(y, 51, 125)

# generate noisy phantom
#print(eng.randn('seed'))
#eng.randn('seed')
#eng.rand('seed')
sigma = sigma/100
if variable_noise == 1:
    map = eng.helper.getNoiseMap(y, noise_factor)
else:
    map = eng.ones(eng.size(y))

eta = eng.times(sigma, map)
if distribution == 'Rice':
    z = eng.sqrt(eng.power(eng.plus(y, eng.times(eta, eng.randn(eng.size(y)))), 2) + eng.power(eng.times(eta, eng.randn(eng.size(y))), 2))
else:
    z = eng.plus(y, eng.times(eta, eng.randn(eng.size(y))))
#print(z)
print('Denoising Started')

[y_est, sigma_est] = eng.bm4d(z, distribution, eng.times(eng.minus(1, estimate_sigma), sigma), profile, do_wiener, verbose, nargout=2)

# plot histogram of the estimated standard deviation
if 'estimate_sigma' in locals():
    eng.helper.visualizeEstMap(y, sigma_est, eta, nargout=0)

eng.helper.visualizeXsect(y, z, y_est, nargout=0)




# 1 - estimate_sigma to change boolean
eng.quit()
