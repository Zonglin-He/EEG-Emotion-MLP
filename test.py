import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from tqdm import tqdm

from model import Network
from data import data_loader


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
        for _, batch in enumerate(valid_loader):
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
           'Num Epoches': 100,
           'Patience': 100,
           }

    # Process each subject[1:]
    subjects = pd.read_csv('data/id_list.csv')['id'].drop_duplicates().tolist()
    for s in subjects:
        print(f"\nProcessing Subject {s}")

        # Load data
        train_loader, valid_loader = data_loader(s)

        # Initialize model and optimizer
        model = Network().to(device)

        model_path = os.path.join(ckpt_dir, f'valence_{s}.pth')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        val_loss, metrics = valid_one_epoch(model, valid_loader, device)


        print(f'=== {s} ===\n')
        mse = metrics['mse']
        mae = metrics['mae']
        cc = metrics['cc']
        print(f'MSE {mse}\n')
        print(f'MAE {mae}\n')
        print(f'CC {cc}\n\n')

