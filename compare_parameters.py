#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter comparison test program for top-down view generation
Generates top-down views with different parameters for the same depth data and compares results
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
from depth_processor.visualization import create_depth_grid_visualization

# Configure matplotlib for English text
setup_matplotlib_english()

def load_depth_grid_from_csv(csv_path):
    """Load compressed depth grid from CSV file"""
    try:
        print(f"Loading CSV file: {csv_path}")
        depth_grid = np.loadtxt(csv_path, delimiter=',')
        print(f"Successfully loaded: shape {depth_grid.shape}")
        return depth_grid
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return None

def generate_point_cloud(depth_grid, scaling_factor=10.0):
    """
    Generate point cloud from depth grid
    
    Args:
        depth_grid: Depth grid data
        scaling_factor: Depth scaling factor
        
    Returns:
        tuple: (point_cloud, absolute_depth_grid)
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
    Generate top-down views with different parameters and compare results
    
    Args:
        csv_path: Path to CSV file
        save_dir: Directory to save results
    """
    # Load CSV file
    depth_grid = load_depth_grid_from_csv(csv_path)
    if depth_grid is None:
        return False
    
    # Generate point cloud
    point_cloud, absolute_depth = generate_point_cloud(depth_grid)
    if point_cloud is None or point_cloud.size == 0:
        print("Failed to generate point cloud")
        return False
    
    print(f"Point cloud generated: {point_cloud.shape[0]} points")
    
    # Point cloud statistics
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    print(f"Point cloud range - X: {x_min:.2f}m - {x_max:.2f}m, Y: {y_min:.2f}m - {y_max:.2f}m, Z: {z_min:.2f}m - {z_max:.2f}m")
    
    # Height distribution
    y_percentiles = np.percentile(point_cloud[:, 1], [5, 25, 50, 75, 95])
    print(f"Height (Y) percentiles [5,25,50,75,95]: {y_percentiles}")
      # Parameter variations
    grid_resolutions = [0.05, 0.1, 0.2]  # meters/cell
    height_thresholds = [0.1, 0.2, 0.3]  # meters
      # Dictionary to store results
    results = {}
    
    # Depth grid visualization (for reference)
    grid_vis = create_depth_grid_visualization(depth_grid, cell_size=30)
    
    # Test all parameter combinations
    for res in grid_resolutions:
        for thresh in height_thresholds:
            print(f"Test: resolution={res}m/cell, height threshold={thresh}m")
            
            # Adjust grid size based on resolution
            grid_size = int(4.0 / res)  # Cover a range of 4m
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
