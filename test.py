import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from ultralytics import YOLO
from dataset import ComparativePoseDataset
# from network import ComparativeTemporalLSTM
from network import ComparativeTemporalCNN
import train


def main():
    parser = argparse.ArgumentParser(description='Test ComparativeTemporalLSTM model')
    parser.add_argument('--batchsize', '-b', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--dataset', '-d', default='data/test', help='Test dataset directory')
    parser.add_argument('--model', '-m', default='model/lstm_comp/best_model.pth', help='Path to the trained model')
    args = parser.parse_args()

    # デバイス設定
    device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")

    # ポーズモデルのロード
    pose_model = YOLO("yolo11n-pose.pt").to(device)

    # データセットのロード
    try:
        dataset = ComparativePoseDataset(args.dataset, pose_model, cache_dir="pose_cache_test")
        dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # モデルのロード
    net = ComparativeTemporalCNN(train.input_dim, train.hidden_dim, train.output_dim, train.kernel_size, train.num_layers).to(device)
    try:
        net.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Model loaded from {args.model}")
    except FileNotFoundError:
        print(f"Error: Model file {args.model} not found.")
        return
    net.eval()  # モデルを評価モードに設定

    # テストループ
    total_absolute_error = 0
    total_squared_error = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, true_score_diff in dataloader:
            inputs, true_score_diff = inputs.to(device), true_score_diff.to(device)

            # 推論
            predicted_score_diff = net(inputs).squeeze()

            # 評価指標の計算
            absolute_error = torch.abs(predicted_score_diff - true_score_diff)
            squared_error = (predicted_score_diff - true_score_diff) ** 2

            total_absolute_error += absolute_error.sum().item()
            total_squared_error += squared_error.sum().item()
            total_samples += len(true_score_diff)

            # 各サンプルの結果を表示
            for true, pred in zip(true_score_diff.cpu().numpy(), predicted_score_diff.cpu().numpy()):
                print(f"True Score Difference: {true:.2f}, Predicted: {pred:.2f}")

    # 平均絶対誤差（MAE）と二乗平均平方根誤差（RMSE）
    mae = total_absolute_error / total_samples
    rmse = (total_squared_error / total_samples) ** 0.5
    print(f"Total Samples: {total_samples}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


if __name__ == '__main__':
    main()
