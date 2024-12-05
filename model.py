import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Normal, StudentT, LogNormal, MixtureSameFamily, Categorical
)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.hierarchical_embedding = nn.ModuleList([
            nn.Linear(input_dim, d_model//2),
            nn.Linear(d_model//2, d_model)
        ])
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x, mask=None):
        for layer in self.hierarchical_embedding:
            x = self.dropout(F.relu(layer(x)))
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.transformer(x, mask=mask)

class ARDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt = self.dropout(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.layer_norm(output)

class NARDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        tgt = self.dropout(tgt)
        output = self.decoder(tgt, memory)
        return self.layer_norm(output)

class AdaptiveDistributionModule(nn.Module):
    def __init__(self, d_model, num_features, num_components=2, dropout=0.2):
        super().__init__()
        self.num_components = num_components
        self.num_features = num_features
        
        # Project concatenated input to correct dimension
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Project concatenated dimension back to d_model
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Parameter networks
        hidden_dim = d_model // 2  # Reduce hidden dimension
        self.weight_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features * num_components)
        )
        
        self.loc_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features * num_components)
        )
        
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features * num_components)
        )
        
        self.df_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features * num_components)
        )

    def forward(self, h_t):
        batch_size, seq_len, _ = h_t.shape
        
        # Project to correct dimension
        h_t = self.projection(h_t)
        
        # Generate distribution parameters
        weights = self.weight_net(h_t)
        weights = weights.view(batch_size, seq_len, self.num_features, self.num_components)
        weights = F.softmax(weights, dim=-1)
        
        locs = self.loc_net(h_t)
        locs = locs.view(batch_size, seq_len, self.num_features, self.num_components)
        
        scales = self.scale_net(h_t)
        scales = scales.view(batch_size, seq_len, self.num_features, self.num_components)
        scales = F.softplus(scales) + 1e-6
        
        dfs = self.df_net(h_t)
        dfs = dfs.view(batch_size, seq_len, self.num_features, self.num_components)
        dfs = F.softplus(dfs) + 2
        
        return weights, locs, scales, dfs


class HYDRA(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=8, num_decoder_layers=8, num_dist_components=2, dropout=0.1):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout2d(dropout/2)
        self.encoder = TransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_encoder_layers, dropout=dropout
        )
        self.ar_decoder = ARDecoder(
            d_model=d_model, nhead=nhead, num_layers=num_decoder_layers, dropout=dropout
        )
        self.nar_decoder = NARDecoder(
            d_model=d_model, nhead=nhead, num_layers=num_decoder_layers, dropout=dropout
        )
        self.dist_module = AdaptiveDistributionModule(
            d_model=d_model, num_features=output_dim, num_components=num_dist_components, dropout=dropout
        )
        self.alpha_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.output_projection = nn.Linear(d_model, output_dim)
        self.target_embedding = nn.Linear(input_dim, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, target_len):
        device = x.device
        batch_size = x.size(0)
        
        # Apply input dropouts
        x = self.input_dropout(x)
        x = self.feature_dropout(x.unsqueeze(-1)).squeeze(-1)
        
        # Generate masks
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(target_len).to(device)
        
        # Encode input sequence
        memory = self.encoder(x, src_mask)
        
        # Prepare target sequence
        tgt = torch.zeros(batch_size, target_len, x.size(-1)).to(device)
        tgt = self.embedding_dropout(self.target_embedding(tgt))
        
        # Decode
        h_ar = self.ar_decoder(tgt, memory, tgt_mask)
        h_nar = self.nar_decoder(tgt, memory)
        
        # Concatenate decoder outputs
        h_combined = torch.cat([h_ar, h_nar], dim=-1)  # Shape: [batch_size, seq_len, d_model*2]
        
        # Generate distribution parameters
        weights, locs, scales, dfs = self.dist_module(h_combined)
        
        # Compute attention weights
        alpha = self.alpha_net(torch.cat([h_ar, h_nar, h_ar], dim=-1))  # Use h_ar instead of z
        
        # Final prediction
        blended = alpha * h_ar + (1 - alpha) * h_nar
        output = self.output_projection(self.output_dropout(blended))
        
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

    def compute_loss(self, pred_dict, target, lambda1=0.4, lambda2=0.4, lambda3=0.2):
        batch_size, seq_len, num_features = target.shape
        
        # Point forecasting loss
        point_loss = F.mse_loss(pred_dict['prediction'], target)
        
        # Distribution loss
        dist_loss = 0.0
        for i in range(num_features):
            mix = Categorical(pred_dict['weights'][..., i, :])
            comp = StudentT(
                pred_dict['dfs'][..., i, :],
                pred_dict['locs'][..., i, :],
                pred_dict['scales'][..., i, :]
            )
            gmm = MixtureSameFamily(mix, comp)
            dist_loss -= gmm.log_prob(target[..., i]).mean()
        dist_loss = dist_loss / num_features
        
        # Uncertainty loss
        pred_errors = F.mse_loss(pred_dict['prediction'], target, reduction='none')
        ar_better = (pred_errors.mean(dim=-1) < pred_errors.mean()).float()
        uncert_loss = F.binary_cross_entropy(pred_dict['alpha'].squeeze(-1), ar_better)
        
        # Combined loss
        total_loss = lambda1 * point_loss + lambda2 * dist_loss + lambda3 * uncert_loss
        
        return total_loss, {
            'point_loss': point_loss.item(),
            'dist_loss': dist_loss.item(),
            'uncert_loss': uncert_loss.item()
        }