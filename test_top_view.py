#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
圧縮深度データからトップダウンビューを生成するテストプログラム
CSVファイルから深度データを読み込み、グリッド表示とトップダウンビューを生成して表示します
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
from depth_processor.visualization import create_depth_grid_visualization, create_default_depth_image

def load_depth_grid_from_csv(csv_path):
    """
    CSVファイルから圧縮された深度グリッドデータを読み込む
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        numpy.ndarray: 読み込まれた深度グリッドデータ
    """
    try:
        print(f"CSVファイルを読み込み中: {csv_path}")
        depth_grid = np.loadtxt(csv_path, delimiter=',')
        print(f"読み込み成功: 形状 {depth_grid.shape}")
        return depth_grid
    except Exception as e:
        print(f"CSVファイルの読み込みに失敗しました: {e}")
        return None

def save_test_csv(file_path, rows=12, cols=16):
    """
    テスト用のCSVファイルを生成する関数（実際のデータがない場合用）
    
    Args:
        file_path: 保存先のパス
        rows: 行数 (デフォルト=12)
        cols: 列数 (デフォルト=16)
    """
    # テスト用の深度データを生成（中心に近いほど値が小さい=近い）
    y, x = np.ogrid[:rows, :cols]
    center_y, center_x = rows // 2, cols // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 0.1から0.9の範囲に正規化（inverted: 中心ほど値が小さくなる）
    max_dist = np.sqrt((rows)**2 + (cols)**2) / 2
    depth_grid = 0.1 + 0.8 * (distance / max_dist)
    
    # 左側に障害物を配置（値を大きくして遠くに）
    obstacle_mask = (x < cols // 3) & (np.abs(y - center_y) < rows // 4)
    depth_grid[obstacle_mask] = 0.05
    
    # 上部に障害物を配置
    top_obstacle = (y < rows // 4) & (x > cols // 3) & (x < 2 * cols // 3)
    depth_grid[top_obstacle] = 0.05
    
    # 一部に無効値（0）を設定
    invalid_mask = (x > 3 * cols // 4) & (y > 3 * rows // 4)
    depth_grid[invalid_mask] = 0
    
    # CSVファイルとして保存
    try:
        np.savetxt(file_path, depth_grid, delimiter=',', fmt='%.4f')
        print(f"テスト用CSVファイルを生成しました: {file_path}")
        return True
    except Exception as e:
        print(f"CSVファイルの生成に失敗しました: {e}")
        return False

def process_depth_grid(depth_grid, save_dir=None):
    """
    圧縮深度グリッドからトップダウンビューを生成して表示する
    
    Args:
        depth_grid: 深度グリッドデータ
        save_dir: 結果を保存するディレクトリ（Noneの場合は保存しない）
    """
    if depth_grid is None:
        print("有効な深度グリッドがありません")
        return False
    
    # パラメータ設定
    grid_rows, grid_cols = depth_grid.shape
    print(f"深度グリッドサイズ: {grid_rows}x{grid_cols}")
    
    # 圧縮データ用のパラメータ
    scaling_factor = 10.0  # 深度スケーリング係数
    grid_resolution = 0.08  # メートル/グリッドセル
    grid_width = 60  # トップダウングリッド幅
    grid_height = 60  # トップダウングリッド高さ
    height_threshold = 0.2  # 高さしきい値
    
    # カメラパラメータ（実際のカメラの値に近づけるための概算値）
    fx = 332.5 / 16 * grid_cols * 0.8  # 視野角を広く取る
    fy = 309.0 / 12 * grid_rows * 0.8
    cx = grid_cols / 2.0
    cy = grid_rows / 2.0
    
    # 1. 深度グリッドの可視化（単純に色で深度を表現）
    print("1. 深度グリッドを可視化しています...")
    grid_vis = create_depth_grid_visualization(depth_grid, cell_size=30)
    
    # 2. 絶対深度に変換
    print("2. 絶対深度に変換しています...")
    absolute_depth = convert_to_absolute_depth(depth_grid, scaling_factor, is_compressed_grid=True)
    print(f"   絶対深度の範囲: {np.min(absolute_depth):.2f}m - {np.max(absolute_depth):.2f}m")
    
    # 3. 点群生成
    print("3. 点群を生成しています...")
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
    
    if point_cloud is None or point_cloud.size == 0:
        print("点群の生成に失敗しました")
        return False
    
    print(f"   生成された点群: {point_cloud.shape[0]}点")
    
    # 点群の統計情報
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    print(f"   点群の範囲 - X: {x_min:.2f}m - {x_max:.2f}m, Y: {y_min:.2f}m - {y_max:.2f}m, Z: {z_min:.2f}m - {z_max:.2f}m")
    
    # Y値（高さ）の分布
    y_percentiles = np.percentile(point_cloud[:, 1], [5, 25, 50, 75, 95])
    print(f"   高さ（Y）パーセンタイル [5,25,50,75,95]: {y_percentiles}")
    
    # 4. 占有グリッド生成
    print("4. 占有グリッドを生成しています...")
    occupancy_grid = create_top_down_occupancy_grid(
        point_cloud,
        grid_resolution=grid_resolution,
        grid_width=grid_width,
        grid_height=grid_height,
        height_threshold=height_threshold
    )
    
    if occupancy_grid is None:
        print("占有グリッドの生成に失敗しました")
        return False
    
    print(f"   占有グリッドサイズ: {occupancy_grid.shape}")
    
    # グリッドの統計情報
    unique_values = np.unique(occupancy_grid)
    print(f"   占有グリッド値: {unique_values}")
    unknown_cells = np.sum(occupancy_grid == 0)
    obstacle_cells = np.sum(occupancy_grid == 1)
    free_cells = np.sum(occupancy_grid == 2)
    total_cells = grid_width * grid_height
    print(f"   未知: {unknown_cells}/{total_cells} ({unknown_cells/total_cells*100:.1f}%)")
    print(f"   障害物: {obstacle_cells}/{total_cells} ({obstacle_cells/total_cells*100:.1f}%)")
    print(f"   通行可能: {free_cells}/{total_cells} ({free_cells/total_cells*100:.1f}%)")
    
    # 5. トップダウンビューの可視化
    print("5. トップダウンビューを可視化しています...")
    topdown_vis = visualize_occupancy_grid(occupancy_grid, scale_factor=6)
    
    # 結果を表示
    plt.figure(figsize=(14, 7))
    
    # 深度グリッド可視化
    plt.subplot(1, 3, 1)
    plt.title("深度グリッド可視化")
    plt.imshow(cv2.cvtColor(grid_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 点群を2Dにプロットしてカラーマップ表示（高さをカラーで表現）
    plt.subplot(1, 3, 2)
    plt.title("点群の上面図 (高さ=色)")
    plt.scatter(point_cloud[:, 0], point_cloud[:, 2], c=point_cloud[:, 1], 
                cmap='jet', s=10, alpha=0.7)
    plt.colorbar(label='高さ (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    
    # トップダウンビュー（占有グリッド）
    plt.subplot(1, 3, 3)
    plt.title("トップダウン占有グリッド")
    plt.imshow(cv2.cvtColor(topdown_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 保存するか表示するか
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(save_dir / f"topdown_test_{timestamp}.png", dpi=150, bbox_inches='tight')
        
        # 個別の画像も保存
        cv2.imwrite(str(save_dir / f"depth_grid_{timestamp}.png"), grid_vis)
        cv2.imwrite(str(save_dir / f"topdown_view_{timestamp}.png"), topdown_vis)
        
        print(f"結果を保存しました: {save_dir}")
    else:
        plt.tight_layout()
        plt.show()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="圧縮深度データからトップダウンビューを生成するテストプログラム")
    parser.add_argument("--csv", help="入力CSVファイルのパス", default=None)
    parser.add_argument("--generate", help="テスト用のCSVを生成する", action="store_true")
    parser.add_argument("--rows", type=int, help="生成するCSVの行数", default=12)
    parser.add_argument("--cols", type=int, help="生成するCSVの列数", default=16)
    parser.add_argument("--save-dir", help="結果を保存するディレクトリ", default=None)
    
    args = parser.parse_args()
    
    # CSVファイルのパス
    csv_path = args.csv
    
    # CSVがない場合はテスト用のCSVを生成
    if csv_path is None or args.generate:
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "test_depth_grid.csv")
        
        # テスト用CSVを生成
        save_test_csv(csv_path, rows=args.rows, cols=args.cols)
    
    # CSVファイルを読み込み
    depth_grid = load_depth_grid_from_csv(csv_path)
    
    # 処理実行
    process_depth_grid(depth_grid, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
