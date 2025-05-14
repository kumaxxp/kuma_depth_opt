#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
トップダウンビュー生成のパラメータ比較テストプログラム
同じ深度データに対して異なるパラメータでトップダウンビューを生成し、結果を比較します
"""

import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 自作モジュールをインポート
from depth_processor import convert_to_absolute_depth, depth_to_point_cloud
from depth_processor import create_top_down_occupancy_grid, visualize_occupancy_grid
from depth_processor.visualization import create_depth_grid_visualization

def load_depth_grid_from_csv(csv_path):
    """CSVファイルから圧縮深度グリッドを読み込む"""
    try:
        print(f"CSVファイルを読み込み中: {csv_path}")
        depth_grid = np.loadtxt(csv_path, delimiter=',')
        print(f"読み込み成功: 形状 {depth_grid.shape}")
        return depth_grid
    except Exception as e:
        print(f"CSVファイルの読み込みに失敗しました: {e}")
        return None

def generate_point_cloud(depth_grid, scaling_factor=10.0):
    """
    深度グリッドから点群を生成
    
    Args:
        depth_grid: 深度グリッドデータ
        scaling_factor: 深度スケーリング係数
        
    Returns:
        tuple: (点群, 絶対深度グリッド)
    """
    grid_rows, grid_cols = depth_grid.shape
    
    # カメラパラメータ（実際のカメラの値に近づけるための概算値）
    fx = 332.5 / 16 * grid_cols * 0.8
    fy = 309.0 / 12 * grid_rows * 0.8
    cx = grid_cols / 2.0
    cy = grid_rows / 2.0
    
    # 絶対深度に変換
    absolute_depth = convert_to_absolute_depth(depth_grid, scaling_factor, is_compressed_grid=True)
    
    # 点群生成
    point_cloud = depth_to_point_cloud(
        absolute_depth,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        is_grid_data=True,
        grid_rows=grid_rows,
        grid_cols=grid_cols
    )
    
    return point_cloud, absolute_depth

def compare_parameters(csv_path, save_dir=None):
    """
    異なるパラメータでトップダウンビューを生成して比較
    
    Args:
        csv_path: CSVファイルのパス
        save_dir: 結果保存先ディレクトリ
    """
    # CSVファイルを読み込む
    depth_grid = load_depth_grid_from_csv(csv_path)
    if depth_grid is None:
        return False
    
    # 点群を生成
    point_cloud, absolute_depth = generate_point_cloud(depth_grid)
    if point_cloud is None or point_cloud.size == 0:
        print("点群の生成に失敗しました")
        return False
    
    print(f"点群生成: {point_cloud.shape[0]}点")
    
    # 点群の統計情報
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    print(f"点群の範囲 - X: {x_min:.2f}m - {x_max:.2f}m, Y: {y_min:.2f}m - {y_max:.2f}m, Z: {z_min:.2f}m - {z_max:.2f}m")
    
    # Y値（高さ）の分布
    y_percentiles = np.percentile(point_cloud[:, 1], [5, 25, 50, 75, 95])
    print(f"高さ（Y）パーセンタイル [5,25,50,75,95]: {y_percentiles}")
    
    # パラメータバリエーション
    grid_resolutions = [0.05, 0.1, 0.2]  # メートル/セル
    height_thresholds = [0.1, 0.2, 0.3]  # メートル
    
    # 結果を格納する辞書
    results = {}
    
    # 深度グリッド可視化（参照用）
    grid_vis = create_depth_grid_visualization(depth_grid, cell_size=30)
    
    # すべてのパラメータの組み合わせをテスト
    for res in grid_resolutions:
        for thresh in height_thresholds:
            print(f"テスト: 解像度={res}m/セル, 高さ閾値={thresh}m")
            
            # グリッドサイズを解像度に応じて調整
            grid_size = int(4.0 / res)  # 4mの範囲をカバー
            grid_width = grid_height = grid_size
            
            # 占有グリッドを生成
            occupancy_grid = create_top_down_occupancy_grid(
                point_cloud,
                grid_resolution=res,
                grid_width=grid_width,
                grid_height=grid_height,
                height_threshold=thresh
            )
            
            # 統計情報
            unknown_cells = np.sum(occupancy_grid == 0)
            obstacle_cells = np.sum(occupancy_grid == 1)
            free_cells = np.sum(occupancy_grid == 2)
            total_cells = grid_width * grid_height
            
            # 可視化
            topdown_vis = visualize_occupancy_grid(occupancy_grid, scale_factor=4)
            
            # 結果を保存
            key = f"res{res}_thresh{thresh}"
            results[key] = {
                "occupancy_grid": occupancy_grid,
                "visualization": topdown_vis,
                "stats": {
                    "unknown": unknown_cells / total_cells,
                    "obstacle": obstacle_cells / total_cells,
                    "free": free_cells / total_cells
                }
            }
    
    # 結果を表示または保存
    fig = plt.figure(figsize=(15, 10))
    
    # 元の深度グリッド
    plt.subplot(3, 4, 1)
    plt.title("入力深度グリッド")
    plt.imshow(cv2.cvtColor(grid_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 点群の上面図
    plt.subplot(3, 4, 2)
    plt.title("点群の上面図")
    plt.scatter(point_cloud[:, 0], point_cloud[:, 2], c=point_cloud[:, 1], 
                cmap='jet', s=10, alpha=0.7)
    plt.colorbar(label='高さ (m)')
    plt.grid(True)
    plt.axis('equal')
    
    # 各パラメータでのトップダウンビューを表示
    pos = 5  # 3x4のプロットで、5番目から開始
    
    for res in grid_resolutions:
        for thresh in height_thresholds:
            key = f"res{res}_thresh{thresh}"
            result = results[key]
            
            plt.subplot(3, 4, pos)
            plt.title(f"解像度: {res}m, 閾値: {thresh}m")
            plt.imshow(cv2.cvtColor(result["visualization"], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            # 統計情報をテキストとして表示
            stats = result["stats"]
            info_text = f"障害物: {stats['obstacle']*100:.1f}%\n" \
                        f"通行可能: {stats['free']*100:.1f}%\n" \
                        f"未知: {stats['unknown']*100:.1f}%"
            plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
                    fontsize=8, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            pos += 1
    
    plt.tight_layout()
    
    # 保存または表示
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = Path(csv_path).stem
        
        # 全体の図を保存
        fig.savefig(save_dir / f"{filename}_comparison_{timestamp}.png", dpi=150, bbox_inches='tight')
        
        # 個別の結果も保存
        for res in grid_resolutions:
            for thresh in height_thresholds:
                key = f"res{res}_thresh{thresh}"
                result = results[key]
                
                param_name = f"{filename}_res{res}_thresh{thresh}_{timestamp}.png"
                cv2.imwrite(str(save_dir / param_name), result["visualization"])
        
        print(f"結果を保存しました: {save_dir}")
    else:
        plt.show()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="トップダウンビューのパラメータ比較テスト")
    parser.add_argument("--csv", required=True, help="入力CSVファイルのパス")
    parser.add_argument("--save-dir", help="結果の保存先ディレクトリ", default=None)
    
    args = parser.parse_args()
    
    # パラメータ比較を実行
    compare_parameters(args.csv, args.save_dir)

if __name__ == "__main__":
    main()
