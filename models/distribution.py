import torch
import torch.nn as nn
import torch.distributions as D

class MixtureDistribution(nn.Module):
    def __init__(self, hidden_dim, num_components=4):
        super().__init__()
        self.num_components = num_components
        
        # Component weight network
        self.weight_net = nn.Linear(hidden_dim, num_components)
        
        # Parameter networks for each distribution type
        self.student_net = nn.Linear(hidden_dim, 3)  # df, loc, scale
        self.lognormal_net = nn.Linear(hidden_dim, 2)  # loc, scale
        self.normal_net = nn.Linear(hidden_dim, 2)  # loc, scale
        self.negbin_net = nn.Linear(hidden_dim, 2)  # total_count, logits
        
    def forward(self, h):
        # Get mixture weights
        weights = torch.softmax(self.weight_net(h), dim=-1)
        
        # Get parameters for each distribution
        student_params = self.student_net(h)
        lognormal_params = self.lognormal_net(h)
        normal_params = self.normal_net(h)
        negbin_params = self.negbin_net(h)
        
        # Create distribution components
        components = [
            D.StudentT(df=student_params[..., 0].exp(), 
                      loc=student_params[..., 1],
                      scale=student_params[..., 2].exp()),
            D.LogNormal(loc=lognormal_params[..., 0],
                       scale=lognormal_params[..., 1].exp()),
            D.Normal(loc=normal_params[..., 0],
                    scale=normal_params[..., 1].exp()),
            D.NegativeBinomial(total_count=negbin_params[..., 0].exp(),
                             logits=negbin_params[..., 1])
        ]
        
        return D.MixtureSameFamily(
            D.Categorical(probs=weights),
            D.Independent(D.Categorical(components), 1)
        )