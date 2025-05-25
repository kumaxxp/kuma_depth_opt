#!/usr/bin/env python3
"""
カメラ座標系の表示テスト
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 英語表示用の設定
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

def _plot_camera_coordinate_system(ax):
    """カメラ座標系とピンホールカメラモデルの可視化"""
    # カメラ原点
    camera_pos = np.array([0, 0, 0])
    
    # 座標軸（カメラ座標系）
    axis_length = 2.0
    
    # X軸: 右方向（赤）
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', 
              arrow_length_ratio=0.1, linewidth=3, label='X: Right')
    
    # Y軸: 下方向（緑）- 表示上では上向きに見えるように負の値を使用
    ax.quiver(0, 0, 0, 0, -axis_length, 0, color='green', 
              arrow_length_ratio=0.1, linewidth=3, label='Y: Down')
    
    # Z軸: 前方向（青）
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', 
              arrow_length_ratio=0.1, linewidth=3, label='Z: Forward')
    
    # ビューアングルを調整：Y軸のマイナス側が上に見えるように
    ax.view_init(elev=0, azim=-90)  # 正面から見て、Y軸マイナスが上
    
    # 画像平面の表示
    image_distance = 1.5
    image_width = 1.2
    image_height = 0.9
    
    # 画像平面の四角形（Y軸の向きに合わせて調整）
    corners = np.array([
        [-image_width/2, image_height/2, image_distance],   # 左上（Y軸マイナス側）
        [image_width/2, image_height/2, image_distance],    # 右上
        [image_width/2, -image_height/2, image_distance],   # 右下（Y軸プラス側）
        [-image_width/2, -image_height/2, image_distance],  # 左下
        [-image_width/2, image_height/2, image_distance]    # 閉じる
    ])
    
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'k-', linewidth=2)
    
    # 光学中心と焦点距離の説明
    ax.scatter(0, 0, image_distance, color='orange', s=100, label='Optical Center (cx,cy)')
    
    # 投影線の例（Y軸の向きに合わせて調整）
    world_points = np.array([
        [0.5, -0.3, 2.5],   # Y座標を反転（上側の点）
        [-0.4, 0.2, 2.8],   # 下側の点
        [0.3, 0.4, 3.0]     # さらに下側の点
    ])
    
    for point in world_points:
        # 3D点
        ax.scatter(point[0], point[1], point[2], color='purple', s=50)
        
        # カメラ中心から3D点への線
        ax.plot([0, point[0]], [0, point[1]], [0, point[2]], 'gray', alpha=0.5)
        
        # 画像平面への投影
        scale = image_distance / point[2]
        proj_x = point[0] * scale
        proj_y = point[1] * scale
        ax.scatter(proj_x, proj_y, image_distance, color='red', s=30)
    
    # ピンホールカメラ方程式の表示
    ax.text2D(0.02, 0.98, 'Pinhole Camera Model:\nX = (u - cx) * Z / fx\nY = (v - cy) * Z / fy\n\nNote: Y-axis points DOWN', 
              transform=ax.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 軸ラベルを明確に
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down, negative upward)')
    ax.set_zlabel('Z (Forward)')
    ax.set_title('Camera Coordinate System\n& Pinhole Model')
    ax.legend(fontsize=8)
    
    # 軸の範囲を調整
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])  # Y軸の範囲はそのまま
    ax.set_zlim([0, 3])

def test_camera_coordinate_system():
    """カメラ座標系のテスト表示"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    _plot_camera_coordinate_system(ax)
    
    # 画像として保存
    plt.savefig('test_camera_coordinate_system.png', dpi=300, bbox_inches='tight')
    print("Test camera coordinate system saved as 'test_camera_coordinate_system.png'")
    
    plt.show()

if __name__ == "__main__":
    test_camera_coordinate_system()
