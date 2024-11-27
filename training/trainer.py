import torch
import torch.nn as nn
import torch.optim as optim

class HYDRATrainer:
    def __init__(self, model, lr=1e-4, weight_decay=0.01):
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    def train_step(self, batch):
        x, y = batch
        
        # Forward pass
        dist = self.model(x)
        
        # Compute losses
        nll_loss = -dist.log_prob(y).mean()
        point_loss = nn.MSELoss()(dist.mean, y)
        
        # Total loss
        loss = nll_loss + point_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'nll_loss': nll_loss.item(),
            'point_loss': point_loss.item(),
            'total_loss': loss.item()
        }