import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from tqdm import tqdm

from model import Network
from data import data_loader


def train_one_epoch(model, train_loader, optimizer, device):
    """
    Train model for one epoch

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Model optimizer
        device: Computing device (CPU/GPU)

    Returns:
        average_loss: Average classification loss for the epoch
    """
    model.train()

    criterion = nn.MSELoss(reduction='mean')
    total_loss = 0.0

    pbar = tqdm(train_loader, desc='Training', leave=True)
    for batch in pbar:
        data, label = batch
        label = label[:, 0:1]
        data, label = data.to(device), label.to(device)

        pred_value = model(data)
        loss = criterion(pred_value, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss/len(train_loader), model


def valid_one_epoch(model, valid_loader, device):
    """
    验证一个epoch并计算所有指标
    """
    model.eval()

    criterion = nn.MSELoss(reduction='mean')
    total_loss = 0.0

    all_predictions = []
    all_label = []

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='Validating'):
            data, label = batch
            label = label[:, 0:1]
            data, label = data.to(device), label.to(device)

            pred_value = model(data)
            loss = criterion(pred_value, label)

            all_predictions.extend(pred_value.detach().cpu().numpy())
            all_label.extend(label.detach().cpu().numpy())

            total_loss += loss.item()

    # 计算指标
    metrics = calculate_regression_metrics(np.array(all_label), np.array(all_predictions))

    return total_loss / len(valid_loader), metrics


def calculate_regression_metrics(y_true, y_pred):
    """
    计算回归评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        dict: 包含MSE、MAE和CC的字典
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    cc = np.corrcoef(y_true[:, 0], y_pred[: ,0])[0, 1]

    return {
        'mse': mse,
        'mae': mae,
        'cc': cc
    }


if __name__ == "__main__":
    modal = 'HR_V'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories for saving models and plots
    ckpt_dir = os.path.join('Checkpoints', modal)
    plot_dir = os.path.join('Summary', modal)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    cfg = {'Learning Rate': 1e-3,
           'Weight Decay': 1e-6,
           'Num Epoches': 200,
           'Patience': 30,
           }

    # Process each subject[1:]
    subjects = pd.read_csv('data/id_list.csv')['id'].drop_duplicates().tolist()
    for s in subjects:
        print(f"\nProcessing Subject {s}")

        # Load data
        train_loader, valid_loader = data_loader(s)

        # Initialize model and optimizer
        model = Network().to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg['Learning Rate'], weight_decay=cfg['Weight Decay'])

        # Training state variables
        best_cc = 0
        patience_counter = 0

        # Training loop
        for epoch in range(cfg['Num Epoches']):
            # Train and validate
            train_loss, model = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, metrics = valid_one_epoch(model, valid_loader, device)

            # if val_loss < best_val_loss:
            if metrics['cc'] > best_cc:
                # Save model checkpoint
                model_path = os.path.join(ckpt_dir, f'valence_{s}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics,
                }, model_path)
                best_cc = metrics['cc']

                patience_counter = 0

            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= cfg['Patience']:
                break

