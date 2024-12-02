import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Normal, 
    StudentT, 
    LogNormal, 
    MixtureSameFamily, 
    Categorical
)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=8, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, x, mask=None):
        x = self.dropout(self.embedding(x))
        return self.transformer(x, mask=mask)

class ARDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt = self.dropout(tgt)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask)

class NARDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt = self.dropout(tgt)
        return self.decoder(tgt, memory)

class AdaptiveDistributionModule(nn.Module):
    def __init__(self, d_model, num_components=3):
        super().__init__()
        self.num_components = num_components
        
        # Keep separate networks for better distribution modeling
        self.projection = nn.Linear(d_model * 2, d_model)
        self.weight_net = nn.Linear(d_model, num_components)
        self.loc_net = nn.Linear(d_model, num_components)
        self.scale_net = nn.Linear(d_model, num_components)
        self.df_net = nn.Linear(d_model, num_components)
        
        # Optimize latent variable generation
        self.z_net = nn.Linear(d_model, d_model * 2)
        
    def forward(self, h_t):
        h_t = self.projection(h_t)
        
        # Compute parameters in parallel but keep separate networks
        weights = F.softmax(self.weight_net(h_t), dim=-1)
        locs = self.loc_net(h_t)
        scales = F.softplus(self.scale_net(h_t))
        dfs = F.softplus(self.df_net(h_t)) + 2
        
        # Efficient latent variable generation
        z_params = self.z_net(h_t)
        z_mu, z_sigma = torch.chunk(z_params, 2, dim=-1)
        z_sigma = F.softplus(z_sigma)
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        
        return weights, locs, scales, dfs, z
    
class AdaptiveDistributionModule(nn.Module):
    def __init__(self, d_model, num_components=3, dropout=0.1):
        super().__init__()
        self.num_components = num_components
        self.dropout = nn.Dropout(dropout)

        self.projection = nn.Linear(d_model * 2, d_model)
        self.weight_net = nn.Linear(d_model, num_components)
        self.loc_net = nn.Linear(d_model, num_components)
        self.scale_net = nn.Linear(d_model, num_components)
        self.df_net = nn.Linear(d_model, num_components)
        self.z_net = nn.Linear(d_model, d_model * 2)

    def forward(self, h_t):
        h_t = self.dropout(self.projection(h_t))

        weights = F.softmax(self.weight_net(h_t), dim=-1)
        locs = self.loc_net(h_t)
        scales = F.softplus(self.scale_net(h_t))
        dfs = F.softplus(self.df_net(h_t)) + 2

        z_params = self.z_net(h_t)
        z_mu, z_sigma = torch.chunk(z_params, 2, dim=-1)
        z_sigma = F.softplus(z_sigma)
        z = z_mu + z_sigma * torch.randn_like(z_mu)

        return weights, locs, scales, dfs, z

class HYDRA(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8,
                 num_encoder_layers=8, num_decoder_layers=4, 
                 num_dist_components=3, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        self.ar_decoder = ARDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

        self.nar_decoder = NARDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

        self.dist_module = AdaptiveDistributionModule(
            d_model=d_model,
            num_components=num_dist_components,
            dropout=dropout
        )

        self.alpha_net = nn.Linear(d_model * 3, 1)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.target_embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target_len):
        device = x.device
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(target_len).to(device)

        memory = self.encoder(x, src_mask)

        batch_size = x.size(0)
        tgt = torch.zeros(batch_size, target_len, x.size(-1)).to(device)
        tgt = self.dropout(self.target_embedding(tgt))

        h_ar = self.ar_decoder(tgt, memory, tgt_mask)
        h_nar = self.nar_decoder(tgt, memory)

        weights, locs, scales, dfs, z = self.dist_module(
            torch.cat([h_ar, h_nar], dim=-1)
        )

        alpha = torch.sigmoid(self.alpha_net(
            torch.cat([h_ar, h_nar, z], dim=-1)
        ))

        blended = alpha * h_ar + (1 - alpha) * h_nar
        output = self.output_projection(self.dropout(blended))

        return {
            'prediction': output,
            'alpha': alpha,
            'weights': weights,
            'locs': locs,
            'scales': scales,
            'dfs': dfs
        }
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask
    
    def compute_loss(self, pred_dict, target, lambda1=1.0, lambda2=1.0, lambda3=1.0):
        # Point forecasting loss
        point_loss = F.mse_loss(pred_dict['prediction'], target)
        
        # Reshape tensors for distribution loss
        batch_size, seq_len, num_features = target.shape
        
        # Reshape the parameters to handle multiple features
        weights = pred_dict['weights'].view(batch_size, seq_len, num_features, -1)
        locs = pred_dict['locs'].view(batch_size, seq_len, num_features, -1)
        scales = pred_dict['scales'].view(batch_size, seq_len, num_features, -1)
        dfs = pred_dict['dfs'].view(batch_size, seq_len, num_features, -1)
        
        # Initialize distribution loss
        dist_loss = 0.0
        
        # Compute distribution loss for each feature
        for i in range(num_features):
            mix = Categorical(weights[..., i, :])
            comp = StudentT(
                dfs[..., i, :],
                locs[..., i, :],
                scales[..., i, :]
            )
            gmm = MixtureSameFamily(mix, comp)
            dist_loss -= gmm.log_prob(target[..., i]).mean()
        
        # Average the distribution loss over features
        dist_loss = dist_loss / num_features
        
        # Uncertainty loss (using alpha values)
        # Note: This is a simplified version - you'll need to define ar_better based on your specific criteria
        ar_better = (
            F.mse_loss(pred_dict['prediction'], target, reduction='none').mean(dim=-1) < 
            torch.mean(F.mse_loss(pred_dict['prediction'], target, reduction='none').mean(dim=-1))
        ).float()
        uncert_loss = F.binary_cross_entropy(pred_dict['alpha'].squeeze(-1), ar_better)
        
        # Combined loss
        total_loss = (
            lambda1 * point_loss + 
            lambda2 * dist_loss + 
            lambda3 * uncert_loss
        )
        
        return total_loss, {
            'point_loss': point_loss.item(),
            'dist_loss': dist_loss.item(),
            'uncert_loss': uncert_loss.item()
        }