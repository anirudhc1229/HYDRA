import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import logging
from pathlib import Path
import math
import time

from model import HYDRA

from torch.distributions import (
    Normal, 
    StudentT, 
    LogNormal, 
    MixtureSameFamily, 
    Categorical
)

def generate_synthetic_data(
    num_series=100,
    seq_length=100,
    num_features=3,
    freq_range=(1, 10),
    noise_level=0.1
):
    """
    Generate synthetic time series with trends, seasonality, and noise
    """
    time = np.arange(seq_length)
    data = []
    
    for _ in range(num_series):
        # Generate multiple seasonal components
        series = np.zeros((seq_length, num_features))
        
        for feature in range(num_features):
            # Add trend
            trend = 0.001 * time * np.random.uniform(-1, 1)
            
            # Add seasonal patterns
            num_patterns = np.random.randint(1, 4)
            seasonal = np.zeros_like(time, dtype=float)
            
            for _ in range(num_patterns):
                freq = np.random.uniform(*freq_range)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                seasonal += amplitude * np.sin(2 * np.pi * freq * time / seq_length + phase)
            
            # Add noise
            noise = np.random.normal(0, noise_level, seq_length)
            
            # Combine components
            series[:, feature] = trend + seasonal + noise
        
        data.append(series)
    
    # Stack all series
    data = np.stack(data, axis=0)
    
    return data

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length, stride=1, normalize=True):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.stride = stride
        
        # Calculate statistics for normalization
        if normalize:
            # Calculate mean and std across all timesteps for each feature
            self.means = np.mean(data, axis=(0, 1))
            self.stds = np.std(data, axis=(0, 1))
            # Avoid division by zero
            self.stds = np.where(self.stds == 0, 1.0, self.stds)
            # Normalize the data
            normalized_data = (data - self.means[None, None, :]) / self.stds[None, None, :]
            self.data = torch.FloatTensor(normalized_data)
        else:
            self.data = torch.FloatTensor(data)
            self.means = None
            self.stds = None
        
    def __len__(self):
        return (len(self.data) * ((self.data.shape[1] - self.seq_length - self.pred_length + 1) // self.stride))
        
    def __getitem__(self, idx):
        # Convert idx to series_idx and start_idx
        series_idx = idx // ((self.data.shape[1] - self.seq_length - self.pred_length + 1) // self.stride)
        start_idx = (idx % ((self.data.shape[1] - self.seq_length - self.pred_length + 1) // self.stride)) * self.stride
        
        x = self.data[series_idx, start_idx:start_idx + self.seq_length]
        y = self.data[series_idx, start_idx + self.seq_length:start_idx + self.seq_length + self.pred_length]
        return x, y
    
    def inverse_transform(self, normalized_data):
        """
        Convert normalized data back to original scale
        """
        if self.means is None or self.stds is None:
            return normalized_data
        
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
            
        return (normalized_data * self.stds[None, None, :]) + self.means[None, None, :]

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hydra_synthetic_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'num_series': 100,
        'seq_length': 96,
        'pred_length': 24,
        'num_features': 3,
        'batch_size': 128,
        'd_model': 256,
        'nhead': 4,
        'num_encoder_layers': 4,
        'num_decoder_layers': 2,
        'num_dist_components': 3,
        'learning_rate': 1e-5,
        'warmup_steps': 5000,
        'num_epochs': 50,
        'stride': 4,  # Add stride parameter to reduce dataset size
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("Generating synthetic data...")

    # Generate synthetic data
    data = generate_synthetic_data(
        num_series=config['num_series'],
        seq_length=1000,  # Longer sequence for splitting
        num_features=config['num_features']
    )
    
    # Split data
    train_ratio, val_ratio = 0.7, 0.15
    n = data.shape[1]
    
    train_data = data[:, :int(n * train_ratio)]
    val_data = data[:, int(n * train_ratio):int(n * (train_ratio + val_ratio))]
    test_data = data[:, int(n * (train_ratio + val_ratio)):]
    
    logger.info(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Create dataloaders with stride and normalization
    train_dataset = TimeSeriesDataset(
        train_data, 
        config['seq_length'], 
        config['pred_length'],
        stride=config['stride'],
        normalize=True  # Enable normalization
    )
    
    # Use same normalization statistics from training set for validation and test
    val_dataset = TimeSeriesDataset(
        val_data, 
        config['seq_length'], 
        config['pred_length'],
        stride=config['stride'],
        normalize=True
    )
    
    test_dataset = TimeSeriesDataset(
        test_data, 
        config['seq_length'], 
        config['pred_length'],
        stride=config['stride'],
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = HYDRA(
        input_dim=config['num_features'],
        output_dim=config['num_features'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        num_dist_components=config['num_dist_components']
    ).to(config['device'])
    
    # Train model
    logger.info("Starting training...")
    model = train_hydra(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        device=config['device'],
        checkpoint_dir='checkpoints/synthetic'
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_metrics(model, test_loader, config['device'])
    logger.info(f"Test metrics: {metrics}")
    
    # Save results
    results = {
        'dataset': 'synthetic',
        'pred_length': config['pred_length'],
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / 'synthetic_results.csv'
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"Results saved to {results_path}")
    
    # Generate and plot some predictions
    logger.info("Generating predictions for visualization...")
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x, y = x.to(config['device']), y.to(config['device'])
        output = model(x, y.size(1))
        
        # Convert predictions and targets back to original scale
        predictions = test_dataset.inverse_transform(output['prediction'].cpu())
        targets = test_dataset.inverse_transform(y.cpu())
        
        # Save predictions
        np.save('results/synthetic_predictions.npy', predictions)
        np.save('results/synthetic_targets.npy', targets)
        
        logger.info("Predictions saved to results/synthetic_predictions.npy")

def train_hydra(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    warmup_steps=5000,
    device='cuda',
    checkpoint_dir='checkpoints'
):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )
    
    logger = logging.getLogger(__name__)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    step = 0
    
    # Calculate total steps for progress reporting
    total_steps = len(train_loader)
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x, y.size(1))
            loss, loss_components = model.compute_loss(output, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step < warmup_steps:
                scheduler.step()
            
            train_losses.append(loss.item())
            step += 1
            
            # Print progress every 10% of epoch or at least every 50 batches
            if batch_idx % max(total_steps // 10, 50) == 0:
                elapsed = time.time() - epoch_start_time
                progress = (batch_idx + 1) / total_steps
                eta = elapsed / progress - elapsed if progress > 0 else 0
                
                logger.info(
                    f'Epoch {epoch}, Progress: {progress:.1%}, '
                    f'Batch {batch_idx}/{total_steps}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Point Loss: {loss_components["point_loss"]:.4f}, '
                    f'Dist Loss: {loss_components["dist_loss"]:.4f}, '
                    f'Uncert Loss: {loss_components["uncert_loss"]:.4f}, '
                    f'ETA: {eta:.0f}s'
                )
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x, y.size(1))
                loss, _ = model.compute_loss(output, y)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f'Epoch {epoch} completed in {epoch_time:.0f}s, '
            f'Train Loss: {avg_train_loss:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}'
        )
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / f'hydra_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
    
    return model

def evaluate_metrics(model, test_loader, device):
    model.eval()
    nmae_list = []
    nrmse_list = []
    crps_list = []
    nll_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, y.size(1))
            pred = output['prediction']
            
            # Compute metrics
            mae = torch.abs(pred - y).mean(dim=1)
            rmse = torch.sqrt(torch.mean((pred - y) ** 2, dim=1))
            
            # Normalize by target scale
            y_std = torch.std(y, dim=1, unbiased=True)
            nmae = (mae / y_std).mean().item()
            nrmse = (rmse / y_std).mean().item()
            
            # Distribution metrics - compute for each feature separately
            batch_size, seq_len, num_features = y.shape
            
            # Reshape distribution parameters
            weights = output['weights'].view(batch_size, seq_len, num_features, -1)
            locs = output['locs'].view(batch_size, seq_len, num_features, -1)
            scales = output['scales'].view(batch_size, seq_len, num_features, -1)
            dfs = output['dfs'].view(batch_size, seq_len, num_features, -1)
            
            # Initialize feature-wise metrics
            batch_nll = 0
            batch_crps = 0
            
            # Compute metrics for each feature
            for i in range(num_features):
                # Create mixture distribution for this feature
                mix = Categorical(weights[..., i, :])
                comp = StudentT(
                    dfs[..., i, :],
                    locs[..., i, :],
                    scales[..., i, :]
                )
                gmm = MixtureSameFamily(mix, comp)
                
                # Compute NLL for this feature
                batch_nll -= gmm.log_prob(y[..., i]).mean().item()
                
                # Sample and compute CRPS for this feature
                samples = gmm.sample((100,))
                batch_crps += compute_crps(samples, y[..., i]).mean().item()
            
            # Average across features
            nll_list.append(batch_nll / num_features)
            crps_list.append(batch_crps / num_features)
            nmae_list.append(nmae)
            nrmse_list.append(nrmse)
    
    metrics = {
        'NMAE': np.mean(nmae_list),
        'NRMSE': np.mean(nrmse_list),
        'CRPS': np.mean(crps_list),
        'NLL': np.mean(nll_list)
    }
    
    return metrics

def compute_crps(samples, target):
    """
    Compute Continuous Ranked Probability Score (CRPS)
    """
    n_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]
    
    # Compute empirical CDF
    positions = torch.arange(1, n_samples + 1).float().to(samples.device)
    positions = positions.unsqueeze(-1).unsqueeze(-1) / n_samples
    
    # Heaviside function for target
    heaviside = (samples >= target.unsqueeze(0)).float()
    
    # Compute CRPS
    crps = ((positions - heaviside) ** 2).mean(dim=0)
    return crps
    return crps

if __name__ == "__main__":
    main()