import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import torch
from torch.distributions import MultivariateNormal
from torchdiffeq import odeint_adjoint

from jko_utils import *

import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample_type', type=str, default="single",choices=["single","group"])

parser.add_argument('--model_id', type=int, default=0)
parser.add_argument('--L', type=int, default=4)

parser.add_argument('--solver', type=str, default="rk4",choices=["rk4","dopri5"])
parser.add_argument('--dt_factor', type=float, default=1.)
parser.add_argument('--forward_flow_idx', type=int, default=124)

parser.set_defaults(feature=False)

args = parser.parse_args()


savepath = "results/flow_model_{}".format(args.model_id)
sample_savepath = "{}/samples".format(savepath)
if not os.path.exists(sample_savepath):
    os.mkdir(sample_savepath)

with open("{}/config".format(savepath), "rb") as f:
    argdict = pickle.load(f)



# load lunar surface data, standardize, apply PCA reduction

x0_img = np.load("dem/lola_240MPP_dem_patches_64.npy")

with open("{}/fit_pca".format(savepath), "rb") as f:
    pca = pickle.load(f)
x0_unstandardized = torch.tensor(pca.transform(x0_img.reshape(x0_img.shape[0], -1)))

x0_mean = x0_unstandardized.mean()
x0_std = x0_unstandardized.std()

x0 = (x0_unstandardized - x0_mean) / x0_std



# load flow nets

t_span = torch.tensor([0.,1.])

L = args.L
L_orig = argdict["L"]
dt = argdict["dt"]

normalize_time = argdict["normalize_time"]

nn_width = argdict["nn_width"]
nn_depth = argdict["nn_depth"]

if L != L_orig:
    print("Using fewer ({} < {}) flow nets than during training!".format(L, L_orig))

flow_nets = []
for k in range(L):
    block_savepath = "{}/block_{}".format(savepath, k+1)
    dynamics_and_div = TemporalNetworkDivergence(x0.shape[-1],nn_width,nn_depth)
    dynamics_and_div.load_state_dict(torch.load("{}/net_state".format(block_savepath)))
    flow_nets.append(dynamics_and_div)


if args.sample_type == "single":
    # Forward-Time Flow (Data -> Noise)

    sampled_x1_traj = []
    with torch.no_grad():
        _xk = x0[args.forward_flow_idx]
        
        initial_condition = torch.cat([_xk.unsqueeze(0).float(), torch.zeros(1).unsqueeze(-1)], dim=-1)
        for k in range(L):
            if normalize_time:
                sub_t_span = torch.tensor([t_span[0], t_span[1]])
            else:
                sub_t_span = torch.tensor([t_span[0] + k/L, t_span[0] + (k+1)/L])
            _tilde_xk1 = odeint_adjoint(flow_nets[k],
                                        initial_condition,
                                        sub_t_span,
                                        method='rk4',
                                        options={'step_size':dt/args.dt_factor})

            sampled_x1_traj.append(_tilde_xk1[-1,0,:-1])
            initial_condition = _tilde_xk1[-1]
    sampled_x1_traj = torch.stack(sampled_x1_traj)

    fig, ax = plt.subplots(1,L+1, figsize=(4*(L+1), 4))

    flow_im = ax[0].imshow(pca.inverse_transform(x0[args.forward_flow_idx]*x0_std + x0_mean).reshape(64,64), cmap='gray')
    cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(flow_im, cax=cax, orientation='vertical')
    ax[0].set_title("Before Flow", fontsize=18)

    for i in range(L):
        flow_im = ax[i+1].imshow(pca.inverse_transform(sampled_x1_traj[i]*x0_std + x0_mean).reshape(64,64), cmap='gray')
        cax = make_axes_locatable(ax[i+1]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(flow_im, cax=cax, orientation='vertical')
        ax[i+1].set_title("Block {}".format(i+1), fontsize=18)

    plt.tight_layout()
    plt.savefig("{}/forward_flow_sample{}_L{}.png".format(sample_savepath, args.forward_flow_idx, L))
    plt.clf()
    plt.close()



    # Reverse-Time Flow (Noise -> Data)

    noise_dist = MultivariateNormal(torch.zeros(x0.shape[-1]), torch.eye(x0.shape[-1]))

    _z = noise_dist.sample([1])

    sampled_x0_traj = []
    with torch.no_grad():
        zk = _z

        initial_condition = torch.cat([zk, torch.zeros(zk.shape[0]).unsqueeze(-1)], dim=-1)
        for k in range(L):
            if normalize_time:
                sub_t_span = torch.tensor([t_span[1], t_span[0]])
            else:
                sub_t_span = torch.tensor([t_span[1] - k/L, t_span[1] - (k+1)/L])

            if args.solver == "rk4":
                tilde_sampled_x0 = odeint_adjoint(flow_nets[L-(k+1)],
                                                  initial_condition,
                                                  sub_t_span,
                                                  method='rk4',
                                                  options={'step_size':dt/args.dt_factor})
            elif args.solver == "dopri5":
                tilde_sampled_x0 = odeint_adjoint(flow_nets[L-(k+1)],
                                                  initial_condition,
                                                  sub_t_span,
                                                  method='dopri5')
            sampled_x0_traj.append(tilde_sampled_x0[-1,:,:-1])
            initial_condition = tilde_sampled_x0[-1]
    sampled_x0_traj = torch.stack(sampled_x0_traj)

    fig, ax = plt.subplots(1,L+1, figsize=(4*(L+1), 4))

    flow_im = ax[0].imshow(pca.inverse_transform(_z*x0_std + x0_mean).reshape(64,64), cmap='gray')
    cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(flow_im, cax=cax, orientation='vertical')
    ax[0].set_title("Noise", fontsize=18)

    for i in range(L):
        flow_im = ax[i+1].imshow(pca.inverse_transform(sampled_x0_traj[i]*x0_std + x0_mean).reshape(64,64), cmap='gray')
        cax = make_axes_locatable(ax[i+1]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(flow_im, cax=cax, orientation='vertical')
        ax[i+1].set_title("Block {}".format(L-i), fontsize=18)

    plt.tight_layout()
    sample_idx = 0
    while os.path.exists("{}/reverse_flow_sample{}_L{}.png".format(sample_savepath, sample_idx, L)):
        sample_idx += 1
    plt.savefig("{}/reverse_flow_sample{}_L{}.png".format(sample_savepath, sample_idx, L))
    plt.clf()
    plt.close()


elif args.sample_type == "group":
    # Forward-Time Flow (Data -> Noise)

    sampled_x1_traj = []
    with torch.no_grad():
        _xk = x0[args.forward_flow_idx:args.forward_flow_idx+5]
        
        initial_condition = torch.cat([_xk.float(), torch.zeros(5).unsqueeze(-1)], dim=-1)
        for k in range(L):
            if normalize_time:
                sub_t_span = torch.tensor([t_span[0], t_span[1]])
            else:
                sub_t_span = torch.tensor([t_span[0] + k/L, t_span[0] + (k+1)/L])
            _tilde_xk1 = odeint_adjoint(flow_nets[k],
                                        initial_condition,
                                        sub_t_span,
                                        method='rk4',
                                        options={'step_size':dt/args.dt_factor})

            sampled_x1_traj.append(_tilde_xk1[-1,:,:-1])
            initial_condition = _tilde_xk1[-1]
    sampled_x1_traj = torch.stack(sampled_x1_traj)

    
    fig, ax = plt.subplots(5,L+1, figsize=(4*(L+1), 4*5))

    for n in range(5):

        flow_im = ax[n,0].imshow(x0_img[args.forward_flow_idx+n], cmap='gray')
        cax = make_axes_locatable(ax[n,0]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(flow_im, cax=cax, orientation='vertical')
        ax[n,0].set_title("Before Flow", fontsize=18)

        for i in range(L):
            flow_im = ax[n,i+1].imshow(pca.inverse_transform(sampled_x1_traj[i,n]*x0_std + x0_mean).reshape(64,64), cmap='gray')
            cax = make_axes_locatable(ax[n,i+1]).append_axes('right', size='5%', pad=0.05)
            fig.colorbar(flow_im, cax=cax, orientation='vertical')
            ax[n,i+1].set_title("Block {}".format(i+1), fontsize=18)

    plt.tight_layout()
    plt.savefig("{}/group_forward_flow_sample{}_L{}.png".format(sample_savepath, args.forward_flow_idx, L))
    plt.clf()
    plt.close()



    # Reverse-Time Flow (Noise -> Data)

    noise_dist = MultivariateNormal(torch.zeros(x0.shape[-1]), torch.eye(x0.shape[-1]))

    _z = noise_dist.sample([5])

    sampled_x0_traj = []
    with torch.no_grad():
        zk = _z

        initial_condition = torch.cat([zk, torch.zeros(zk.shape[0]).unsqueeze(-1)], dim=-1)
        for k in range(L):
            if normalize_time:
                sub_t_span = torch.tensor([t_span[1], t_span[0]])
            else:
                sub_t_span = torch.tensor([t_span[1] - k/L, t_span[1] - (k+1)/L])

            if args.solver == "rk4":
                tilde_sampled_x0 = odeint_adjoint(flow_nets[L-(k+1)],
                                                  initial_condition,
                                                  sub_t_span,
                                                  method='rk4',
                                                  options={'step_size':dt/args.dt_factor})
            elif args.solver == "dopri5":
                tilde_sampled_x0 = odeint_adjoint(flow_nets[L-(k+1)],
                                                  initial_condition,
                                                  sub_t_span,
                                                  method='dopri5')
            sampled_x0_traj.append(tilde_sampled_x0[-1,:,:-1])
            initial_condition = tilde_sampled_x0[-1]
    sampled_x0_traj = torch.stack(sampled_x0_traj)


    fig, ax = plt.subplots(5,L+1, figsize=(4*(L+1), 4*5))

    for n in range(5):

        flow_im = ax[n,0].imshow(pca.inverse_transform(_z[n]*x0_std + x0_mean).reshape(64,64), cmap='gray')
        cax = make_axes_locatable(ax[n,0]).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(flow_im, cax=cax, orientation='vertical')
        ax[n,0].set_title("Noise", fontsize=18)

        for i in range(L):
            flow_im = ax[n,i+1].imshow(pca.inverse_transform(sampled_x0_traj[i,n]*x0_std + x0_mean).reshape(64,64), cmap='gray')
            cax = make_axes_locatable(ax[n,i+1]).append_axes('right', size='5%', pad=0.05)
            fig.colorbar(flow_im, cax=cax, orientation='vertical')
            ax[n,i+1].set_title("Block {}".format(L-i), fontsize=18)

    plt.tight_layout()
    sample_idx = 0
    while os.path.exists("{}/group_reverse_flow_sample{}_L{}.png".format(sample_savepath, sample_idx, L)):
        sample_idx += 1
    plt.savefig("{}/group_reverse_flow_sample{}_L{}.png".format(sample_savepath, sample_idx, L))
    plt.clf()
    plt.close()