import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from ultralytics import YOLO
from dataset import ComparativePoseDataset
# from network import ComparativeTemporalLSTM
from network import ComparativeTemporalCNN
import random

# ハイパーパラメータ
input_dim = 26  # Number of keypoints * 2 (x, y)
hidden_dim = 3036
output_dim = 1  # Predict a single score
kernel_size = 10
num_layers = 2
learning_rate = 0.001
model_directory = 'model/lstm_comp'


def set_seed(seed):
    """再現性のためにシード値を固定"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train ComparativeTemporalLSTM')
    parser.add_argument('--batchsize', '-b', type=int, default=10, help='Batch size')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--dataset', '-d', default='data/train', help='Dataset directory')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # シード値を固定
    set_seed(args.seed)

    # デバイス設定
    device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")

    # ポーズモデルのロード
    pose_model = YOLO("yolo11x-pose.pt").to(device)

    # データセットのロード
    try:
        dataset = ComparativePoseDataset(args.dataset, pose_model)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # データセットを分割 (80%訓練, 20%検証)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # データローダーを作成
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

    # モデル、オプティマイザ、損失関数
    net = ComparativeTemporalCNN(input_dim, hidden_dim, output_dim, kernel_size, num_layers).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    os.makedirs(model_directory, exist_ok=True)

    # 学習ループ
    best_val_loss = float('inf')  # 最良の検証損失値
    for epoch in range(args.epoch):
        # 訓練フェーズ
        net.train()
        running_loss = 0.0
        for inputs, score_diff in train_loader:
            inputs, score_diff = inputs.to(device), score_diff.to(device)

            # 順伝播
            optimizer.zero_grad()
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, score_diff)

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 訓練損失の計算
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epoch}, Training Loss: {avg_train_loss:.4f}")

        # 検証フェーズ
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, score_diff in val_loader:
                inputs, score_diff = inputs.to(device), score_diff.to(device)
                outputs = net(inputs).squeeze()
                loss = criterion(outputs, score_diff)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{args.epoch}, Validation Loss: {avg_val_loss:.4f}")

        # ベストモデルの保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), f"{model_directory}/best_model.pth")
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

        # 定期保存
        torch.save(net.state_dict(), f"{model_directory}/model_epoch_{epoch + 1}.pth")

    # 最終モデルの保存
    torch.save(net.state_dict(), f"{model_directory}/model_final.pth")
    print("Training completed. Final model saved.")


if __name__ == '__main__':
    main()
