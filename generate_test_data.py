#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
異なるシナリオのテスト用深度グリッドCSVファイルを生成するスクリプト
"""

import os
import numpy as np
import argparse
from pathlib import Path

def generate_empty_room_grid(rows=12, cols=16):
    """平坦な床面を持つ空の部屋のテストデータ"""
    # 基本的な深度値（床）
    depth_grid = np.ones((rows, cols)) * 0.7
    
    # 少し距離変化をつける（手前から奥に向かって緩やかに遠くなる）
    y_indices = np.arange(rows).reshape(-1, 1)
    gradient = 0.2 * (y_indices / rows)
    depth_grid += gradient
    
    return depth_grid

def generate_obstacle_grid(rows=12, cols=16):
    """障害物がある部屋のテストデータ"""
    # 基本的な深度値（床）
    depth_grid = np.ones((rows, cols)) * 0.7
    
    # 少し距離変化をつける
    y_indices = np.arange(rows).reshape(-1, 1)
    gradient = 0.2 * (y_indices / rows)
    depth_grid += gradient
    
    # 中央に障害物を配置（値を小さくして近くに）
    center_y, center_x = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    obstacle_mask = ((x - center_x)**2 + (y - center_y)**2 < min(rows, cols) // 4)
    depth_grid[obstacle_mask] = 0.2
    
    return depth_grid

def generate_corridor_grid(rows=12, cols=16):
    """廊下のようなシーンのテストデータ"""
    # 基本的な深度値（床）
    depth_grid = np.ones((rows, cols)) * 0.7
    
    # 廊下の奥行き方向の勾配
    y_indices = np.arange(rows).reshape(-1, 1)
    gradient = 0.3 * (y_indices / rows)
    depth_grid += gradient
    
    # 左右の壁を配置
    wall_width = cols // 4
    depth_grid[:, :wall_width] = 0.2  # 左壁
    depth_grid[:, -wall_width:] = 0.2  # 右壁
    
    return depth_grid

def generate_stairs_grid(rows=12, cols=16):
    """階段のようなシーンのテストデータ"""
    # 基本的な深度値
    depth_grid = np.ones((rows, cols)) * 0.7
    
    # 階段のステップ（高さの変化）
    steps = 4
    step_size = rows // steps
    
    for i in range(steps):
        start_row = i * step_size
        end_row = (i+1) * step_size
        if end_row > rows:
            end_row = rows
        
        # 各ステップごとに深度を変える（遠くなるほど値が大きい=遠い）
        depth_grid[start_row:end_row, :] = 0.5 + 0.1 * i
    
    return depth_grid

def generate_complex_scene_grid(rows=12, cols=16):
    """複雑なシーンのテストデータ（床、壁、複数の障害物）"""
    # 基本的な床面
    depth_grid = np.ones((rows, cols)) * 0.7
    
    # 床の勾配
    y_indices = np.arange(rows).reshape(-1, 1)
    gradient = 0.2 * (y_indices / rows)
    depth_grid += gradient
    
    # 複数の障害物を配置
    # 障害物1: 左上
    y1, x1 = rows // 4, cols // 4
    radius1 = min(rows, cols) // 8
    y, x = np.ogrid[:rows, :cols]
    mask1 = ((x - x1)**2 + (y - y1)**2 < radius1**2)
    depth_grid[mask1] = 0.2
    
    # 障害物2: 右下
    y2, x2 = 3 * rows // 4, 3 * cols // 4
    radius2 = min(rows, cols) // 10
    mask2 = ((x - x2)**2 + (y - y2)**2 < radius2**2)
    depth_grid[mask2] = 0.3
    
    # 壁: 奥側
    wall_depth = rows // 6
    depth_grid[:wall_depth, :] = 0.15
    
    return depth_grid

def generate_all_test_csvs(output_dir):
    """すべてのテストパターンのCSVを生成"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 標準サイズ
    rows, cols = 12, 16
    
    # 高解像度サイズ（オプション）
    hd_rows, hd_cols = 24, 32
    
    # 各シナリオのグリッドを生成して保存
    scenarios = {
        "empty_room": generate_empty_room_grid,
        "obstacle": generate_obstacle_grid,
        "corridor": generate_corridor_grid,
        "stairs": generate_stairs_grid,
        "complex_scene": generate_complex_scene_grid
    }
    
    for name, generator in scenarios.items():
        # 標準解像度
        grid = generator(rows, cols)
        file_path = output_dir / f"{name}_grid.csv"
        np.savetxt(file_path, grid, delimiter=',', fmt='%.4f')
        print(f"生成: {file_path}")
        
        # 高解像度（オプション）
        hd_grid = generator(hd_rows, hd_cols)
        hd_file_path = output_dir / f"{name}_grid_hd.csv"
        np.savetxt(hd_file_path, hd_grid, delimiter=',', fmt='%.4f')
        print(f"生成: {hd_file_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="異なるシナリオのテスト用深度グリッドCSVを生成")
    parser.add_argument("--output-dir", default="test_data", help="出力ディレクトリ")
    args = parser.parse_args()
    
    # 出力ディレクトリの絶対パスを取得
    if os.path.isabs(args.output_dir):
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    
    generate_all_test_csvs(output_dir)
    print(f"すべてのテストCSVファイルが {output_dir} に生成されました")

if __name__ == "__main__":
    main()
