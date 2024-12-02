import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.hydra import HYDRA
from data.monash_loader import MonashDataset
from training.trainer import HYDRATrainer

def plot_forecasts(context, target, prediction, title):
    plt.figure(figsize=(12, 6))
    
    x_context = np.arange(len(context))
    plt.plot(x_context, context, 'b-', label='Context', alpha=0.5)
    
    x_target = np.arange(len(context), len(context) + len(target))
    plt.plot(x_target, target, 'g-', label='Target', alpha=0.5)
    plt.plot(x_target, prediction, 'r--', label='Prediction')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model, test_loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0
    
    forecasts = []
    actuals = []
    contexts = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            dist = model(batch['context'])
            prediction = dist.mean
            
            mse = torch.mean((prediction - batch['target'])**2)
            mae = torch.mean(torch.abs(prediction - batch['target']))
            
            total_mse += mse.item() * len(batch['target'])
            total_mae += mae.item() * len(batch['target'])
            num_samples += len(batch['target'])
            
            forecasts.append(prediction.cpu().numpy())
            actuals.append(batch['target'].cpu().numpy())
            contexts.append(batch['context'].cpu().numpy())
    
    return {
        'RMSE': np.sqrt(total_mse / num_samples),
        'MAE': total_mae / num_samples,
        'forecasts': forecasts,
        'actuals': actuals,
        'contexts': contexts
    }

def main():
    # Initialize model
    model = HYDRA(
        input_dim=1,
        hidden_dim=512,
        num_heads=8,
        num_layers=8,
        dropout=0.1
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = MonashDataset(split="train")
    test_dataset = MonashDataset(split="test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32
    )
    
    # Initialize trainer
    trainer = HYDRATrainer(model)
    
    # Training loop
    num_epochs = 10
    print("Starting training...")
    
    for epoch in range(num_epochs):
        train_metrics = []
        for batch in train_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            metrics = trainer.train_step(batch)
            train_metrics.append(metrics)
            
        # Print epoch metrics
        avg_metrics = {
            k: sum(m[k] for m in train_metrics) / len(train_metrics)
            for k in train_metrics[0].keys()
        }
        print(f"Epoch {epoch}: {avg_metrics}")
        
        # Evaluate and visualize every 5 epochs
        if epoch % 5 == 0:
            print("\nEvaluating on test set...")
            test_metrics = evaluate_model(model, test_loader, model.device)
            print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
            print(f"Test MAE: {test_metrics['MAE']:.4f}\n")
            
            # Visualize some examples
            for i in range(min(3, len(test_metrics['forecasts']))):
                plot_forecasts(
                    test_metrics['contexts'][i][0],
                    test_metrics['actuals'][i][0],
                    test_metrics['forecasts'][i][0],
                    f'Forecast Example {i+1}'
                )

if __name__ == "__main__":
    main()