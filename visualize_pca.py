import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pca_dim', type=int, default=128)
parser.add_argument('--patch_idx', type=int, default=124)

args = parser.parse_args()


# load lunar surface data, apply PCA reduction

x0_img = np.load("dem/lola_240MPP_dem_patches_64.npy")

pca = PCA(n_components=args.pca_dim)
x0_unstandardized = pca.fit_transform(x0_img.reshape(x0_img.shape[0], -1))
x0_orig = pca.inverse_transform(x0_unstandardized)


# plot pca reconstruction

fig, ax = plt.subplots(1,2)

raw_im = ax[0].imshow(x0_img[args.patch_idx], cmap='gray')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(raw_im, cax=cax, orientation='vertical')
ax[0].set_title("Raw")

mapped_im = ax[1].imshow(x0_orig.reshape(-1, 64, 64)[args.patch_idx], cmap='gray')
cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(mapped_im, cax=cax, orientation='vertical')
ax[1].set_title("PCA Reconstruction")

plt.tight_layout()
plt.savefig("results/pca_reconstruction_{}d_patch{}.png".format(args.pca_dim,args.patch_idx))
plt.clf()
plt.close()