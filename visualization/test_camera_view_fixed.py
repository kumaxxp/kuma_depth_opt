#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 英語表示用の設定
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

def test_camera_coordinate_system():
    """修正されたカメラ座標系の単独テスト"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # カメラ原点
    camera_pos = np.array([0, 0, 0])
    
    # 座標軸（カメラ座標系）
    axis_length = 2.0
    
    # X軸: 右方向（赤）
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', 
              arrow_length_ratio=0.1, linewidth=3, label='X: Right')
    
    # Y軸: 下方向（緑）
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', 
              arrow_length_ratio=0.1, linewidth=3, label='Y: Down')
    
    # Z軸: 前方向（青）
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', 
              arrow_length_ratio=0.1, linewidth=3, label='Z: Forward')
    
    # ビューアングルを調整：Z軸を奥行方向、Y軸を縦方向に
    ax.view_init(elev=-30, azim=-90-70, roll=0+80)  # 少し上から斜めに見下ろす角度
    
    # 画像平面の表示
    image_distance = 1.5
    image_width = 1.2
    image_height = 0.9
    
    # 画像平面の四角形（カメラ座標系に合わせて調整）
    corners = np.array([
        [-image_width/2, -image_height/2, image_distance],   # 左上
        [image_width/2, -image_height/2, image_distance],    # 右上
        [image_width/2, image_height/2, image_distance],     # 右下
        [-image_width/2, image_height/2, image_distance],    # 左下
        [-image_width/2, -image_height/2, image_distance]    # 閉じる
    ])
    
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'k-', linewidth=2, label='Image Plane')
    
    # 光学中心と焦点距離の説明
    ax.scatter(0, 0, image_distance, color='orange', s=100, label='Optical Center (cx,cy)')
    
    # 投影線の例（標準的なカメラ座標系）
    world_points = np.array([
        [0.5, 0.3, 2.5],    # 右上の点
        [-0.4, -0.2, 2.8],  # 左下の点
        [0.3, -0.4, 3.0]    # 右下の点
    ])
    
    for i, point in enumerate(world_points):
        # 3D点
        ax.scatter(point[0], point[1], point[2], color='purple', s=50)
        
        # カメラ中心から3D点への線
        ax.plot([0, point[0]], [0, point[1]], [0, point[2]], 'gray', alpha=0.5)
        
        # 画像平面への投影
        scale = image_distance / point[2]
        proj_x = point[0] * scale
        proj_y = point[1] * scale
        ax.scatter(proj_x, proj_y, image_distance, color='red', s=30)
        
        # 点にラベルを追加
        ax.text(point[0], point[1], point[2], f'P{i+1}', fontsize=8)
    
    # ピンホールカメラ方程式の表示
    ax.text2D(0.02, 0.98, 'Pinhole Camera Model:\nX = (u - cx) * Z / fx\nY = (v - cy) * Z / fy\n\nCoordinate System:\nX: Right, Y: Down, Z: Forward', 
              transform=ax.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 軸ラベルを明確に
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Forward into scene)')
    ax.set_title('Camera Coordinate System\n& Pinhole Model (Fixed View)')
    ax.legend(fontsize=8)
    
    # 軸の範囲を調整
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 3])
    
    # 軸の比率を等しく保つ
    ax.set_box_aspect([1,1,1.5])  # Z軸を少し長めに表示
    
    # 追加の視覚的ガイド
    # カメラ位置を強調
    ax.scatter(0, 0, 0, color='black', s=200, marker='s', label='Camera Origin')
    
    # 座標系の説明を追加
    ax.text2D(0.02, 0.02, 'View: Z-axis goes into the scene (depth)\nY-axis points downward\nX-axis points rightward', 
              transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('test_camera_coordinate_system_fixed.png', dpi=300, bbox_inches='tight')
    print("Test visualization saved as 'test_camera_coordinate_system_fixed.png'")
    plt.show()

if __name__ == "__main__":
    test_camera_coordinate_system()
