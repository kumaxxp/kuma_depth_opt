import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

def visualize_coordinate_systems():
    """
    カメラ座標系とトップダウンビューの座標系の関係を3Dで可視化します。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # カメラの位置と姿勢を定義
    camera_position = np.array([0, 0, 0])  # カメラ原点
    
    # 座標軸の長さ
    axis_length = 2.0
    grid_size = 6  # トップダウングリッドのサイズ
    
    # カメラ座標系の軸を描画（赤=X軸、緑=Y軸、青=Z軸）
    # X軸: 右方向
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
              axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='Camera X (右)')
    # Y軸: 下方向
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
              0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Camera Y (下)')
    # Z軸: 前方向
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], 
              0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Camera Z (前)')
    
    # カメラスクリーンを表現（カメラ前方に配置）
    screen_distance = 1.0  # カメラからスクリーンまでの距離
    screen_width = 1.6
    screen_height = 1.2
    
    # スクリーンの角の座標
    corners = np.array([
        [-screen_width/2, -screen_height/2, screen_distance],  # 左上
        [screen_width/2, -screen_height/2, screen_distance],   # 右上
        [screen_width/2, screen_height/2, screen_distance],    # 右下
        [-screen_width/2, screen_height/2, screen_distance],   # 左下
    ])
    
    # スクリーンを透明な面として描画
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    verts = [corners]
    collection = Poly3DCollection(verts, alpha=0.3, color='cyan')
    ax.add_collection3d(collection)
    
    # カメラスクリーンのラベル
    ax.text(0, 0, screen_distance, "Camera Screen", color='black')
    
    # トップダウングリッドを表現（Y=-1.5の高さに配置）
    grid_height = -1.5  # Y軸の値（床の高さ）
    grid_center = np.array([0, grid_height, grid_size/2])  # グリッドの中心（カメラの正面）
    
    # グリッドの端の座標（X-Z平面上）
    grid_corners = np.array([
        [-grid_size/2, grid_height, 0],             # 左手前
        [grid_size/2, grid_height, 0],              # 右手前
        [grid_size/2, grid_height, grid_size],      # 右奥
        [-grid_size/2, grid_height, grid_size],     # 左奥
    ])
    
    # グリッドを透明な面として描画
    grid_verts = [grid_corners]
    grid_collection = Poly3DCollection(grid_verts, alpha=0.3, color='lightgreen')
    ax.add_collection3d(grid_collection)
    
    # グリッド座標系の軸を描画
    # グリッドX軸: カメラのX軸に一致
    ax.quiver(grid_center[0], grid_center[1], grid_center[2], 
              axis_length, 0, 0, color='darkred', arrow_length_ratio=0.1, label='Grid X')
    # グリッドY軸: カメラのZ軸方向
    ax.quiver(grid_center[0], grid_center[1], grid_center[2], 
              0, 0, axis_length, color='darkblue', arrow_length_ratio=0.1, label='Grid Y')
    
    # トップダウングリッドのラベル
    ax.text(0, grid_height, grid_size/2, "Top-down Grid", color='black')
    
    # カメラからグリッドへの投影線を描画（ポイントクラウドへの変換をイメージ）
    projection_points = np.array([
        [-0.5, -0.5, 2.0],  # 左上の点
        [0.5, -0.5, 2.0],   # 右上の点
        [0.5, 0.5, 2.0],    # 右下の点
        [-0.5, 0.5, 2.0],   # 左下の点
    ])
    
    # 投影点をグリッドにマッピング
    for point in projection_points:
        # カメラ原点から各点への線
        ax.plot([camera_position[0], point[0]], 
                [camera_position[1], point[1]], 
                [camera_position[2], point[2]], 'k--', alpha=0.5)
        
        # 各点からグリッドへの垂直投影
        grid_point_x = point[0]
        grid_point_z = point[2]
        ax.plot([point[0], grid_point_x], 
                [point[1], grid_height], 
                [point[2], grid_point_z], 'k:', alpha=0.5)
        
        # 投影点をマーク
        ax.scatter(grid_point_x, grid_height, grid_point_z, color='blue', marker='o')
    
    # 3D点群の例をプロット（ランダムな点）
    np.random.seed(42)
    n_points = 50
    pc_x = np.random.uniform(-1.0, 1.0, n_points)
    pc_y = np.random.uniform(-0.5, 0.5, n_points)
    pc_z = np.random.uniform(1.5, 3.5, n_points)
    
    ax.scatter(pc_x, pc_y, pc_z, color='purple', alpha=0.5, label='Point Cloud')
    
    # 点群からグリッドへの投影例（いくつかの点のみ）
    for i in range(0, n_points, 10):
        ax.plot([pc_x[i], pc_x[i]], 
                [pc_y[i], grid_height], 
                [pc_z[i], pc_z[i]], 'r:', alpha=0.3)
        
        # 投影点をマーク
        ax.scatter(pc_x[i], grid_height, pc_z[i], color='red', marker='x')
    
    # グラフの設定
    ax.set_xlabel('X軸')
    ax.set_ylabel('Y軸')
    ax.set_zlabel('Z軸')
    ax.set_title('カメラ座標系とトップダウンビューの関係')
    
    # カメラから見た視点を調整（トップダウングリッドが水平に見えるように）
    ax.view_init(elev=0, azim=180, roll=-90)
    ax.set_box_aspect([1,1,1])  # 軸比率を均等に

    # 軸の範囲を調整
    ax.set_xlim([-grid_size/2-1, grid_size/2+1])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, grid_size+1])
    
    # 凡例の表示
    ax.legend()
    
    # グリッド表示
    ax.grid(True)
    
    # 座標変換の説明テキスト
    fig.text(0.02, 0.02, '''座標変換関係:
1. カメラ座標系: X(右), Y(下), Z(前)
2. トップダウングリッド: 
   • グリッドX = カメラX
   • グリッドY = カメラZ
   • 高さ = カメラY
   • 原点はカメラから見て前方''', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_coordinate_systems()