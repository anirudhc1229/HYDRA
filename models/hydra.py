import torch
import torch.nn as nn
from .decoder import ARDecoder, NARDecoder
from .distribution import MixtureDistribution

class HYDRA(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=512,
                 num_heads=8,
                 num_layers=8,
                 dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        
        # Decoders
        self.ar_decoder = ARDecoder(hidden_dim, num_heads, num_layers)
        self.nar_decoder = NARDecoder(hidden_dim, num_heads, num_layers)
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Linear(hidden_dim, 1)
        
        # Distribution modeling
        self.distribution = MixtureDistribution(hidden_dim)
        
        # Blend weights
        self.blend_net = nn.Linear(hidden_dim * 2 + 1, 1)
        
    def forward(self, x, padding_mask=None):
        # Embed input
        h = self.input_embed(x)
        
        # Get decoder outputs
        h_ar = self.ar_decoder(h, padding_mask)
        h_nar = self.nar_decoder(h, padding_mask)
        
        # Compute uncertainty score
        uncertainty = self.uncertainty_net(torch.abs(h_ar - h_nar))
        
        # Compute blend weights
        alpha = torch.sigmoid(self.blend_net(
            torch.cat([h_ar, h_nar, uncertainty], dim=-1)
        ))
        
        # Blend representations
        h_blend = alpha * h_ar + (1 - alpha) * h_nar
        
        # Get distribution parameters
        return self.distribution(h_blend)