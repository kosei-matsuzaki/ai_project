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
    # pose_model = YOLO("yolo11n-pose.pt").to(device)
    pose_model = YOLO("yolo11x-pose.pt").to(device)
    
    # データセットのロード
    try:
        dataset = ComparativePoseDataset(args.dataset, pose_model)
        dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Extract score differences
    score_diffs = [score_diff for _, _, score_diff in dataset.pairs]
    print(score_diffs)

    # Calculate normalization parameters
    score_mean = np.mean(score_diffs)
    score_std = np.std(score_diffs)

    # Normalize score differences

    # モデル、オプティマイザ、損失関数
    # net = ComparativeTemporalCNN(input_dim, hidden_dim, output_dim, num_layers, bidirectional=True).to(device)
    net = ComparativeTemporalCNN(input_dim, hidden_dim, output_dim, kernel_size, num_layers).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    os.makedirs(model_directory, exist_ok=True)

    # 学習ループ
    best_loss = float('inf')  # 最良の損失値
    for epoch in range(args.epoch):
        running_loss = 0.0
        net.train()  # モデルを訓練モードに設定
        for i, (inputs, score_diff) in enumerate(dataloader):
            inputs, score_diff = inputs.to(device), score_diff.to(device)

            # normalized_score_diff = (score_diff - score_mean) / score_std

            # 順伝播
            optimizer.zero_grad()
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, score_diff)

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ミニバッチごとのログ出力
            # if (i + 1) % 10 == 0:
                # print(f"[Epoch {epoch + 1}/{args.epoch}, Batch {i + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        # エポックごとのログ出力
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epoch}, Average Loss: {avg_loss:.4f}")

        # ベストモデルの保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), f"{model_directory}/best_model.pth")
            print(f"Best model saved with loss: {best_loss:.4f}")

        # 定期保存
        torch.save(net.state_dict(), f"{model_directory}/model_epoch_{epoch + 1}.pth")

    # 最終モデルの保存
    torch.save(net.state_dict(), f"{model_directory}/model_final.pth")
    print("Training completed. Final model saved.")


if __name__ == '__main__':
    main()
