#!/usr/bin/env python3
"""
Grid Compression Mapping をテストするためのスクリプト
高解像度設定でのハイライト表示を確認
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def test_grid_compression(grid_cols=16, grid_rows=12):
    """Grid Compression Mappingのテスト"""
    original_width, original_height = 640, 480
    cell_width = original_width // grid_cols
    cell_height = original_height // grid_rows
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # グリッドの描画
    ax.set_xlim(0, original_width)
    ax.set_ylim(original_height, 0)  # 画像座標系（上が原点）
    
    # グリッドライン
    for i in range(grid_rows + 1):
        y = i * cell_height
        ax.axhline(y=y, color='blue', linewidth=1)
    
    for j in range(grid_cols + 1):
        x = j * cell_width
        ax.axvline(x=x, color='blue', linewidth=1)
    
    # ハイライトセルの配置
    if grid_rows >= 3 and grid_cols >= 3:
        highlight_cells = [(1, 1), (grid_rows//2, grid_cols//2)]
        if grid_rows > 3 and grid_cols > 3:
            highlight_cells.append((grid_rows-2, grid_cols-2))
    else:
        highlight_cells = [(0, 1), (1, 0)]
        if grid_rows > 2 and grid_cols > 2:
            highlight_cells.append((grid_rows-1, grid_cols-1))
    
    colors = ['red', 'green', 'orange']
    
    print(f"Grid size: {grid_cols}x{grid_rows}")
    print(f"Cell size: {cell_width}x{cell_height} pixels")
    print(f"Highlight cells: {highlight_cells}")
    
    for idx, (row, col) in enumerate(highlight_cells[:len(colors)]):
        if row < grid_rows and col < grid_cols:
            x_start = col * cell_width
            y_start = row * cell_height
            
            print(f"Cell ({row},{col}): position ({x_start},{y_start}) to ({x_start+cell_width},{y_start+cell_height})")
            
            # セルの塗りつぶし
            rect = Rectangle((x_start, y_start), cell_width, cell_height,
                            facecolor=colors[idx], alpha=0.3, edgecolor=colors[idx], linewidth=2)
            ax.add_patch(rect)
            
            # セル中心の計算と表示
            center_u = x_start + cell_width * 0.5
            center_v = y_start + cell_height * 0.5
            ax.plot(center_u, center_v, 'ko', markersize=8)
            
            # テキスト位置の調整
            text_offset_v = max(15, cell_height * 0.2)  # セルサイズに応じて調整
            
            # グリッド座標の表示
            ax.text(center_u, center_v - text_offset_v, f'Grid({row},{col})', 
                    ha='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            # 元画像座標の表示
            ax.text(center_u, center_v + text_offset_v, f'Image({center_u:.1f},{center_v:.1f})', 
                    ha='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
      # タイトルと軸ラベル
    ax.set_xlabel('Image X (u)')
    ax.set_ylabel('Image Y (v)')
    ax.set_title(f'Grid Compression Test: {grid_cols}x{grid_rows} Grid')
    
    # 情報表示
    compression_ratio = (grid_cols * grid_rows) / (original_width * original_height)
    info_text = (f'Original: {original_width}x{original_height} = {original_width*original_height:,} pixels\n'
                f'Compressed: {grid_cols}x{grid_rows} = {grid_cols*grid_rows} cells\n'
                f'Cell Size: {cell_width}x{cell_height} pixels\n'
                f'Compression Ratio: {compression_ratio:.6f}')
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            verticalalignment='bottom')
    
    plt.tight_layout()
    filename = f'test_grid_compression_{grid_cols}x{grid_rows}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Test visualization saved as '{filename}'")
    plt.close()

if __name__ == "__main__":
    # 高解像度設定をテスト
    test_grid_compression(16, 12)
    # 低解像度設定もテスト
    test_grid_compression(4, 3)
