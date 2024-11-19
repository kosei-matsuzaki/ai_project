import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import random

class ComparativePoseDataset(Dataset):
    def __init__(self, root_dir, model, clip_duration=4, cache_dir="pose_cache"):
        """
        Args:
            root_dir (str): ルートディレクトリ（大会ごとのディレクトリを含む）
            model (YOLO): ポーズ推定に使用するYOLOモデル
            clip_duration (int): クリップの長さ（秒）
            cache_dir (str): キャッシュディレクトリ
        """
        self.pairs = []  # (基準クリップ, 比較クリップ, スコア差)
        self.model = model
        self.clip_duration = clip_duration
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 各コンテストディレクトリを処理
        for contest in os.listdir(root_dir):
            contest_dir = os.path.join(root_dir, contest)
            if not os.path.isdir(contest_dir):
                continue

            # Contest-specific cache directory
            contest_cache_dir = os.path.join(self.cache_dir, contest)
            os.makedirs(contest_cache_dir, exist_ok=True)

            video_paths = sorted(
                [os.path.join(contest_dir, fname) for fname in os.listdir(contest_dir) if fname.endswith('.mp4')]
            )

            # 動画ごとのスコアを取得
            scores = [self._extract_score(video_path) for video_path in video_paths]

            # 動画ごとのポーズクリップを取得
            clip_features = {video_path: self._load_or_compute_pose(video_path, contest_cache_dir)
                             for video_path in video_paths}

            # 基準動画と比較動画のすべてのペアを作成
            for i, (base_video, base_clips) in enumerate(clip_features.items()):
                base_score = scores[i]
                for j, (eval_video, eval_clips) in enumerate(clip_features.items()):
                    if i == j:  # 同じ動画はペアにしない
                        continue
                    eval_score = scores[j]

                    # スコア差を計算
                    score_diff = eval_score - base_score

                    # 各クリップのペアを保存
                    for base_clip, eval_clip in zip(base_clips, eval_clips):
                        # print("base: ", base_clip, ", eval: ", eval_clip, ", score: score_diff")
                        self.pairs.append((base_clip, eval_clip, score_diff))

    def _extract_score(self, video_path):
        """Extract the score from the filename."""
        fname = os.path.basename(video_path)  # Get the filename
        fname_without_ext = os.path.splitext(fname)[0]  # Remove the extension
        try:
            parts = fname_without_ext.split('_')  # Split by underscores
            if len(parts) < 2:
                raise ValueError(f"Invalid filename format: {fname}. Expected format: <label>_<score>_<optional>.mp4")
            score = float(parts[1])  # Extract the second part as the score
            return score
        except (IndexError, ValueError):
            raise ValueError(f"Invalid filename format: {fname}. Expected format: <label>_<score>_<optional>.mp4")

    def _load_or_compute_pose(self, video_path, contest_cache_dir):
        """キャッシュされたポーズデータをロードするか、新たに計算"""
        video_cache_dir = os.path.join(contest_cache_dir, os.path.basename(video_path))
        os.makedirs(video_cache_dir, exist_ok=True)

        if not os.listdir(video_cache_dir):  # キャッシュがない場合
            print(f"Processing video for pose cache: {video_path}")
            all_clips = self._compute_pose_features(video_path)
            for i, clip in enumerate(all_clips):
                np.save(os.path.join(video_cache_dir, f"clip_{i}.npy"), clip)

        return [np.load(os.path.join(video_cache_dir, f)) for f in sorted(os.listdir(video_cache_dir))]

    def _compute_pose_features(self, video_path):
        """ビデオからポーズデータを計算（1秒間に6フレーム抽出）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        num_frames_per_second = 6  # 1秒間に抽出するフレーム数
        num_frames = self.clip_duration * num_frames_per_second  # 1クリップ内の総フレーム数

        poses_all_clips = []
        default_pose = np.zeros(26)  # デフォルトポーズ（キーポイントが検出されない場合）

        # 各クリップの開始時間（1秒刻みでスライド）
        start_times = np.arange(0, total_duration - self.clip_duration + 1, 1)
        for start_time in start_times:
            poses = []
            for i in range(num_frames):
                # フレーム間隔を計算して取得
                frame_time = start_time + (i / num_frames_per_second)  # 時間単位（秒）
                cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)  # ミリ秒単位で設定
                ret, frame = cap.read()
                if not ret:
                    poses.append(default_pose)
                    continue

                # YOLOモデルでポーズ推定
                results = self.model(frame)
                if results and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints[0].xyn.cpu().numpy()
                    if keypoints.shape[0] == 17:  # 17キーポイントが検出された場合
                        filtered_keypoints = np.delete(keypoints, [1, 2, 3, 4], axis=0).flatten()
                        poses.append(filtered_keypoints)
                    else:
                        poses.append(default_pose)
                else:
                    poses.append(default_pose)

            poses_all_clips.append(np.array(poses, dtype=np.float32))

        cap.release()
        return poses_all_clips

    def __getitem__(self, idx):
        base_clip, eval_clip, score_diff = self.pairs[idx]
        combined_clip = np.concatenate([base_clip, eval_clip], axis=-1)
        return torch.tensor(combined_clip, dtype=torch.float32), torch.tensor(score_diff, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)
