from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pca_dim', type=int, default=128)

args = parser.parse_args()


if not os.path.exists("results"):
    os.mkdir("results")


# load raw DEM
im = Image.open('dem/LDEM_60S_240MPP_ADJ.tiff')
dem_LOLA = np.array(im)


# cut into patches, subtract local mean from each patch
patch_size = 64
dem_patches = []
for i in range(patch_size, dem_LOLA.shape[0], patch_size):
    for j in range(patch_size, dem_LOLA.shape[1], patch_size):
        this_patch = np.copy(dem_LOLA[i-patch_size:i,j-patch_size:j])
        this_patch -= this_patch.mean()
        dem_patches.append(this_patch)
dem_patches = np.array(dem_patches)


# compute median peak-to-peak of all patches
all_ptp = []
for i in range(dem_patches.shape[0]):
    all_ptp.append(np.ptp(dem_patches[i]))
ptp_quantile = np.quantile(all_ptp, 0.5)



# include only patches with less than the median peak-to-peak
retained_dem_patches = []
for i in range(dem_patches.shape[0]):
    if all_ptp[i] < ptp_quantile:
        retained_dem_patches.append(dem_patches[i])
retained_dem_patches = np.array(retained_dem_patches)

np.save("dem/lola_240MPP_dem_patches_{}.npy".format(patch_size),retained_dem_patches)



# plot singular spectrum of patch data
_, S, _ = np.linalg.svd(retained_dem_patches.reshape(retained_dem_patches.shape[0], -1))

plt.plot(S, c='k')
plt.axvline(args.pca_dim,c='r', label=r'$d=$' + str(args.pca_dim))
plt.yscale('log')
plt.xlabel("Dimension", fontsize=16)
plt.ylabel("Singular Value", fontsize=16)
plt.legend(fontsize=16)
plt.savefig("results/spectrum_patches_{}.png".format(patch_size))
plt.clf()
plt.close()