import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint

from jko_utils import *

import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pca_dim', type=int, default=128)

parser.add_argument('--L', type=int, default=4)
parser.add_argument('--h', type=float, default=5e0)
parser.add_argument('--dt', type=float, default=0.05)

parser.add_argument('--normalize_time', action='store_true')

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=5e-4)

parser.add_argument('--lr_decay', type=float, default=1.0)

parser.add_argument('--nn_width', type=int, default=512)
parser.add_argument('--nn_depth', type=int, default=2) # num of hidden layers

parser.add_argument('--plot_freq', type=int, default=10)
parser.add_argument('--report_freq', type=int, default=1)

parser.set_defaults(feature=False)

args = parser.parse_args()



if not os.path.exists("results"):
    os.mkdir("results")
savepath = "results/flow_model_0"
if not os.path.exists(savepath):
    os.mkdir(savepath)
else:
    model_idx = 1
    while True:
        if not os.path.exists("results/flow_model_{}".format(model_idx)):
            break
        model_idx += 1
    savepath = "results/flow_model_{}".format(model_idx)
    os.mkdir(savepath)


argdict = vars(args)
with open("{}/config".format(savepath), "wb") as f:
    pickle.dump(argdict, f)



# load lunar surface data, standardize, apply PCA reduction

x0_img = np.load("dem/lola_240MPP_dem_patches_64.npy")

pca = PCA(n_components=args.pca_dim)
x0_unstandardized = torch.tensor(pca.fit_transform(x0_img.reshape(x0_img.shape[0], -1)))
x0_mean = x0_unstandardized.mean()
x0_std = x0_unstandardized.std()

x0 = (x0_unstandardized - x0_mean) / x0_std

x0_orig = pca.inverse_transform(x0_unstandardized)

with open("{}/fit_pca".format(savepath), "wb") as f:
    pickle.dump(pca, f)



# plot pca reconstruction

fig, ax = plt.subplots(1,2)

raw_im = ax[0].imshow(x0_img[124])
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(raw_im, cax=cax, orientation='vertical')
ax[0].set_title("Raw")

mapped_im = ax[1].imshow(x0_orig.reshape(-1, 64, 64)[124])
cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(mapped_im, cax=cax, orientation='vertical')
ax[1].set_title("PCA Reconstruction")

fig.suptitle("Raw vs PCA Reconstruction")
plt.tight_layout()
plt.savefig("{}/pca_reconstruction.png".format(savepath))
plt.clf()
plt.close()



# train the flow nets

t_span = torch.tensor([0.,1.])

L = args.L
h = args.h
dt = args.dt

normalize_time = args.normalize_time

epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

nn_width = args.nn_width
nn_depth = args.nn_depth


xk = x0

flow_nets = []
flow_net_losses = []
for k in range(L):
    print("Block {}".format(k+1))

    block_savepath = "{}/block_{}".format(savepath, k+1)
    if not os.path.exists(block_savepath):
        os.mkdir(block_savepath)
    
    if normalize_time:
        sub_t_span = torch.tensor([t_span[0], t_span[1]])
    else:
        sub_t_span = torch.tensor([t_span[0] + k/L, t_span[0] + (k+1)/L])
    
    dynamics_and_div = TemporalNetworkDivergence(x0.shape[-1],nn_width,nn_depth)

    optimizer = optim.Adam(dynamics_and_div.parameters(), lr=lr if k==0 else lr*args.lr_decay) # decay learning rate (only once) after block 1

    losses = []
    traj_V = []
    traj_div = []
    traj_W = []
    for epoch in range(epochs):
        xk_batches = get_batches(xk, batch_size=batch_size)
        batch_losses = []
        batch_V = []
        batch_div = []
        batch_W = []
        for xk_batch in xk_batches:
            optimizer.zero_grad()

            V, divergence_int, W = jko_loss(dynamics_and_div, x_t0=xk_batch.float(), t0=sub_t_span[0], t1=sub_t_span[1], dt=dt, h=h)

            loss = torch.mean(V + divergence_int + W)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_V.append(torch.mean(V).item())
            batch_div.append(torch.mean(divergence_int).item())
            batch_W.append(torch.mean(W).item())
        losses.append(np.sum(batch_losses[-1]))
        traj_V.append(np.sum(batch_V[-1]))
        traj_div.append(-1*np.sum(batch_div[-1]))
        traj_W.append(np.sum(batch_W[-1]))

        if epoch!= 0 and (epoch % args.plot_freq == 0):
            plt.plot(losses, color='k')
            plt.title("Loss", fontsize=16)
            plt.savefig("{}/loss.png".format(block_savepath))
            plt.clf()
            plt.close()

            plt.plot(traj_V, color='k')
            plt.title("V Term", fontsize=16)
            plt.savefig("{}/traj_V.png".format(block_savepath))
            plt.clf()
            plt.close()

            plt.plot(traj_div, color='k')
            plt.title("Divergence Term", fontsize=16)
            plt.savefig("{}/traj_div.png".format(block_savepath))
            plt.clf()
            plt.close()

            plt.plot(traj_W, color='k')
            plt.title("Wasserstein Term", fontsize=16)
            plt.savefig("{}/traj_W.png".format(block_savepath))
            plt.clf()
            plt.close()

            with torch.no_grad():
                
                plot_idx = np.random.randint(xk.shape[0])

                initial_cond = torch.cat([xk[plot_idx].unsqueeze(0).float(), torch.zeros(1).unsqueeze(-1)], dim=-1)
                tilde_xk1 = odeint_adjoint(dynamics_and_div,
                                          initial_cond,
                                          sub_t_span,
                                          method='rk4',
                                          options={'step_size':dt})

                xk1 = tilde_xk1[-1,0,:-1]
                
                
                fig, ax = plt.subplots(1,2)
                
                orig_im = ax[0].imshow(pca.inverse_transform(x0[plot_idx]*x0_std + x0_mean).reshape(64,64), cmap='gray')
                cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
                fig.colorbar(orig_im, cax=cax, orientation='vertical')
                ax[0].set_title("Original Data")
                
                flow_im = ax[1].imshow(pca.inverse_transform(xk1*x0_std + x0_mean).reshape(64,64), cmap='gray')
                cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.05)
                fig.colorbar(flow_im, cax=cax, orientation='vertical')
                ax[1].set_title("Forward-Time Flow")
                
                fig.suptitle("Block {} Epoch {}".format(k+1,epoch))
                plt.tight_layout()
                plt.savefig("{}/forward_epoch_{}.png".format(block_savepath, epoch))
                plt.clf()
                plt.close()
                
        if epoch % args.plot_freq == 0:
            print("Epoch {} | Loss: {:.2e} | V: {:.2e} | div f: {:.2e} | W: {:.2e}".format(epoch, losses[-1], traj_V[-1], traj_div[-1], traj_W[-1]))
            
    with torch.no_grad():
                
        plot_idx = np.random.randint(xk.shape[0])

        initial_cond = torch.cat([xk[plot_idx].unsqueeze(0).float(), torch.zeros(1).unsqueeze(-1)], dim=-1)
        tilde_xk1 = odeint_adjoint(dynamics_and_div,
                                  initial_cond,
                                   sub_t_span,
                                  method='rk4',
                                  options={'step_size':dt})

        xk1 = tilde_xk1[-1,0,:-1]


        fig, ax = plt.subplots(1,2)

        orig_im = ax[0].imshow(pca.inverse_transform(x0[plot_idx]*x0_std + x0_mean).reshape(64,64), cmap='gray')
        cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(orig_im, cax=cax, orientation='vertical')
        ax[0].set_title("Original Data")

        flow_im = ax[1].imshow(pca.inverse_transform(xk1*x0_std + x0_mean).reshape(64,64), cmap='gray')
        cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(flow_im, cax=cax, orientation='vertical')
        ax[1].set_title("Forward-Time Flow")

        fig.suptitle("Block {} Final".format(k+1))
        plt.tight_layout()
        plt.savefig("{}/forward_final.png".format(block_savepath))
        plt.clf()
        plt.close()
                
    plt.plot(losses, color='k')
    plt.title("Loss", fontsize=16)
    plt.savefig("{}/loss.png".format(block_savepath))
    plt.clf()
    plt.close()

    plt.plot(traj_V, color='k')
    plt.title("V Term", fontsize=16)
    plt.savefig("{}/traj_V.png".format(block_savepath))
    plt.clf()
    plt.close()

    plt.plot(traj_div, color='k')
    plt.title("Divergence Term", fontsize=16)
    plt.savefig("{}/traj_div.png".format(block_savepath))
    plt.clf()
    plt.close()

    plt.plot(traj_W, color='k')
    plt.title("Wasserstein Term", fontsize=16)
    plt.savefig("{}/traj_W.png".format(block_savepath))
    plt.clf()
    plt.close()
    
    with torch.no_grad():
        initial_cond = torch.cat([xk.float(), torch.zeros(xk.shape[0]).unsqueeze(-1)], dim=-1)
        tilde_xk = odeint_adjoint(dynamics_and_div,
                                  initial_cond,
                                  sub_t_span,
                                  method='rk4',
                                  options={'step_size':dt})

        xk = tilde_xk[-1,:,:-1]
    
    flow_nets.append(dynamics_and_div)
    flow_net_losses.append(losses)

    torch.save(dynamics_and_div.state_dict(), "{}/net_state".format(block_savepath))