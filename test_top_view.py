#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test program for top-down view generation from compressed depth data
Loads depth data from CSV files and generates grid display and top-down view
"""

import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import English text utilities
from english_text_utils import setup_matplotlib_english

# Import custom modules
from depth_processor import convert_to_absolute_depth, depth_to_point_cloud
from depth_processor import create_top_down_occupancy_grid, visualize_occupancy_grid
from depth_processor.visualization import create_depth_grid_visualization, create_default_depth_image

# Configure matplotlib for English/Japanese text (auto-detect)
setup_matplotlib_english()

def load_depth_grid_from_csv(csv_path):
    """
    Load compressed depth grid data from CSV file
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        numpy.ndarray: Loaded depth grid data
    """
    try:
        print(f"Loading CSV file: {csv_path}")
        depth_grid = np.loadtxt(csv_path, delimiter=',')
        print(f"Successfully loaded: shape {depth_grid.shape}")
        return depth_grid
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return None

def save_test_csv(file_path, rows=12, cols=16):
    """
    Generate a test CSV file (for cases where real data is not available)
    
    Args:
        file_path: Path to save the file
        rows: Number of rows (default=12)
        cols: Number of columns (default=16)
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
      # Save as CSV file
    try:
        np.savetxt(file_path, depth_grid, delimiter=',', fmt='%.4f')
        print(f"Generated test CSV file: {file_path}")
        return True
    except Exception as e:
        print(f"Failed to generate CSV file: {e}")
        return False

def process_depth_grid(depth_grid, save_dir=None):
    """
    Generate a top-down view from compressed depth grid and display it
    
    Args:
        depth_grid: Depth grid data
        save_dir: Directory to save results (None if not saving)
    """
    if depth_grid is None:
        print("No valid depth grid available")
        return False
    
    # Parameter setup
    grid_rows, grid_cols = depth_grid.shape
    print(f"Depth grid size: {grid_rows}x{grid_cols}")
    
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
      # 1. Visualize depth grid (simply represent depth with color)
    print("1. Visualizing depth grid...")
    grid_vis = create_depth_grid_visualization(depth_grid, cell_size=30)
    
    # 2. Convert to absolute depth
    print("2. Converting to absolute depth...")
    absolute_depth = convert_to_absolute_depth(depth_grid, scaling_factor, is_compressed_grid=True)
    print(f"   Absolute depth range: {np.min(absolute_depth):.2f}m - {np.max(absolute_depth):.2f}m")
    
    # 3. Generate point cloud
    print("3. Generating point cloud...")
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
        print("Failed to generate point cloud")
        return False
    
    print(f"   Generated point cloud: {point_cloud.shape[0]} points")
    
    # Point cloud statistics
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    print(f"   Point cloud range - X: {x_min:.2f}m - {x_max:.2f}m, Y: {y_min:.2f}m - {y_max:.2f}m, Z: {z_min:.2f}m - {z_max:.2f}m")
    
    # Y value (height) distribution
    y_percentiles = np.percentile(point_cloud[:, 1], [5, 25, 50, 75, 95])
    print(f"   Height (Y) percentiles [5,25,50,75,95]: {y_percentiles}")
    
    # 4. Generate occupancy grid
    print("4. Creating occupancy grid...")
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
    
    # Depth grid visualization
    plt.subplot(1, 3, 1)
    plt.title("Depth Grid Visualization")
    plt.imshow(cv2.cvtColor(grid_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Point cloud 2D plot (height as color)
    plt.subplot(1, 3, 2)
    plt.title("Point Cloud Top View (Height=Color)")
    plt.scatter(point_cloud[:, 0], point_cloud[:, 2], c=point_cloud[:, 1], 
                cmap='jet', s=10, alpha=0.7)
    plt.colorbar(label='Height (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')

    # Top-down view (occupancy grid)
    plt.subplot(1, 3, 3)
    plt.title("Top-Down Occupancy Grid")
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
