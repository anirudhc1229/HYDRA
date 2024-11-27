import torch
import torch.nn as nn

class ARDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4*hidden_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.causal_mask = torch.triu(torch.ones(1024, 1024), diagonal=1).bool()
        
    def forward(self, x, padding_mask=None):
        # Apply causal masking for AR decoding
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len].to(x.device)
        return self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)

class NARDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4*hidden_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
    def forward(self, x, padding_mask=None):
        # NAR decoding uses no masking
        return self.transformer(x, src_key_padding_mask=padding_mask)