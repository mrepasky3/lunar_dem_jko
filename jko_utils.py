import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

def get_batches(data, batch_size):
    idx = torch.randperm(data.shape[0])
    batches = [data[idx[i*batch_size:(i+1)*batch_size]] for i in range(idx.shape[0] // batch_size)]
    return batches

def jko_loss(dynamics_and_div, x_t0, t0, t1, dt, h):
    
    t_span = torch.tensor([t0,t1])
    
    initial_cond = torch.cat([x_t0, torch.zeros(x_t0.shape[0]).unsqueeze(-1)], dim=-1)
    tilde_x_t1 = odeint_adjoint(dynamics_and_div,
                              initial_cond,
                              t_span,
                              method='rk4',
                              options={'step_size':dt})
    
    x_t1 = tilde_x_t1[-1,:,:-1]
    divergence_int = tilde_x_t1[-1,:,-1]
    
    V = torch.pow(torch.linalg.norm(x_t1,ord=2,dim=-1),2)/2 
    W = (1/(2*h)) * torch.pow(torch.linalg.norm(x_t1 - x_t0, ord=2, dim=-1), 2)
    
    return V, divergence_int, W


class TemporalNetworkDivergence(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, time_dep=False):

        super(TemporalNetworkDivergence, self).__init__()

        self.time_dep = time_dep

        layers = []
        if self.time_dep:
            layers.append(nn.Linear(in_features=input_dim+1, out_features=hidden_dim))
        else:
            layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim))
        layers.append(nn.Softplus(beta=20))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(nn.Softplus(beta=20))

        layers.append(nn.Linear(in_features=hidden_dim, out_features=input_dim))

        self.ff = nn.Sequential(*layers)
        
        for m in self.ff.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def pushforward(self, t, x):
        if self.time_dep:
            return self.ff(torch.cat([t.repeat(x.shape[:-1]+(1,)),x],dim=-1))
        else:
            return self.ff(x)
        
    def forward(self, t, x):
        fx = self.pushforward(t, x[:,:-1])
        
        eps = torch.randn_like(x[:,:-1])
        eps_fx = self.pushforward(t, x[:,:-1]+eps)
        
        div_fx = torch.bmm(eps.unsqueeze(-2), eps_fx.unsqueeze(-1)).squeeze(-1)
        
        return torch.cat([fx,-div_fx],dim=-1)
