import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return x, y

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
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    step = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
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
            
            if batch_idx % 100 == 0:
                logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Point Loss: {loss_components["point_loss"]:.4f}, '
                    f'Dist Loss: {loss_components["dist_loss"]:.4f}, '
                    f'Uncert Loss: {loss_components["uncert_loss"]:.4f}'
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
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f'Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}')
        
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
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f'hydra_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
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
            
            # Distribution metrics
            mix = Categorical(output['weights'])
            comp = StudentT(
                output['dfs'],
                output['locs'],
                output['scales']
            )
            )
            gmm = MixtureSameFamily(mix, comp)
            
            nll = -gmm.log_prob(y).mean().item()
            
            # Approximate CRPS using samples
            samples = gmm.sample((100,))
            crps = compute_crps(samples, y).mean().item()
            
            nmae_list.append(nmae)
            nrmse_list.append(nrmse)
            crps_list.append(crps)
            nll_list.append(nll)
    
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

def prepare_monash_data(data_path, dataset_name):
    """
    Load and prepare data from Monash Time Series Repository
    """
    df = pd.read_csv(f"{data_path}/{dataset_name}.csv")
    
    # Convert to numpy array and handle missing values
    data = df.values
    data = np.nan_to_interpolate(data, method='linear')
    
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Split into train, validation, and test sets
    train_ratio, val_ratio = 0.7, 0.15
    n = len(data_scaled)
    
    train_data = data_scaled[:int(n * train_ratio)]
    val_data = data_scaled[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
    test_data = data_scaled[int(n * (train_ratio + val_ratio)):]
    
    return train_data, val_data, test_data, scaler

def create_dataloaders(train_data, val_data, test_data, seq_length, pred_length, batch_size):
    """
    Create DataLoader objects for training, validation, and testing
    """
    train_dataset = TimeSeriesDataset(train_data, seq_length, pred_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length, pred_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length, pred_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hydra_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'data_path': 'path/to/monash/datasets',
        'datasets': [
            'energy_1',
            'finance_1',
            'traffic_1',
            'weather_1'
        ],
        'seq_length': 96,
        'pred_lengths': [24, 48, 96, 192, 336],
        'batch_size': 512,
        'd_model': 256,
        'nhead': 4,
        'num_encoder_layers': 4,
        'num_decoder_layers': 2,
        'num_dist_components': 3,
        'learning_rate': 1e-4,
        'warmup_steps': 1000,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Train and evaluate on each dataset and prediction length
    for dataset_name in config['datasets']:
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Load and prepare data
        train_data, val_data, test_data, scaler = prepare_monash_data(
            config['data_path'],
            dataset_name
        )
        
        for pred_length in config['pred_lengths']:
            logger.info(f"Training for prediction length: {pred_length}")
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_data,
                val_data,
                test_data,
                config['seq_length'],
                pred_length,
                config['batch_size']
            )
            
            # Initialize model
            model = HYDRA(
                input_dim=train_data.shape[1],
                output_dim=train_data.shape[1],
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_encoder_layers=config['num_encoder_layers'],
                num_decoder_layers=config['num_decoder_layers'],
                num_dist_components=config['num_dist_components']
            ).to(config['device'])
            
            # Train model
            model = train_hydra(
                model,
                train_loader,
                val_loader,
                num_epochs=config['num_epochs'],
                learning_rate=config['learning_rate'],
                warmup_steps=config['warmup_steps'],
                device=config['device'],
                checkpoint_dir=f'checkpoints/{dataset_name}/h{pred_length}'
            )
            
            # Evaluate model
            metrics = evaluate_metrics(model, test_loader, config['device'])
            
            # Save results
            results = {
                'dataset': dataset_name,
                'pred_length': pred_length,
                **metrics
            }
            
            results_df = pd.DataFrame([results])
            results_path = results_dir / f'{dataset_name}_h{pred_length}_results.csv'
            results_df.to_csv(results_path, index=False)
            
            logger.info(f"Results saved to {results_path}")
            logger.info(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()