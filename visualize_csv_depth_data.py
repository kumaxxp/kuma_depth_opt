\
import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib # 追加
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # Poly3DCollectionをインポート
import os
import sys

# 日本語フォントの設定 (WSLにインストールしたフォント名に合わせてください)
#matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
# または 'IPAPGothic', 'IPAexGothic' など、インストールしたフォントを指定

# プロジェクトルートをPythonパスに追加するための処理
# このスクリプトがプロジェクトルート直下にあることを想定
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from depth_processor.point_cloud import depth_to_point_cloud
from depth_processor.depth_model import convert_to_absolute_depth

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Config loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        sys.exit(1)

def load_depth_data_from_csv(csv_path):
    """CSVファイルから深度データを読み込む"""
    try:
        depth_data = np.loadtxt(csv_path, delimiter=',')
        print(f"Depth data loaded from {csv_path}, shape: {depth_data.shape}")
        return depth_data
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        sys.exit(1)

def visualize_point_cloud_3d(points, camera_params_config, depth_grid_shape): # depth_grid_shape を追加
    """3Dポイントクラウドとカメラ座標系、各種平面をMatplotlibで可視化する"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # カメラ座標系の軸を描画
    axis_length = 0.5
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='Camera X (右)')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Camera Y (下)')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Camera Z (前)')

    # カメラスクリーンを表現 (coordinate_systems.py を参考に値を調整)
    screen_distance = camera_params_config.get("screen_distance", 0.3) # Z軸方向の距離
    # screen_width と screen_height はポイントクラウドの視野角からある程度推測するか、固定値
    # ここではCSVの解像度と焦点距離から簡易的に画角を考慮してみる (オプション)
    # fx = camera_params_config.get("fx_scaled") # スケーリングされたfx
    # grid_cols = depth_grid_shape[1]
    # if fx and grid_cols > 0:
    #     # 簡易的に、スクリーン距離での横幅を計算 (2 * dist * tan(fov_x/2))
    #     # tan(fov_x/2) = (grid_cols/2) / fx
    #     screen_width_derived = screen_distance * grid_cols / fx if fx != 0 else 0.8
    # else:
    #     screen_width_derived = 0.8
    # screen_height_derived = screen_width_derived * (depth_grid_shape[0] / depth_grid_shape[1]) if depth_grid_shape[1] > 0 else 0.6
    
    # 固定値を使用 (coordinate_systems.py よりスケールを小さく)
    screen_width = 0.6 # camera_params_config.get("screen_width", screen_width_derived)
    screen_height = 0.4 # camera_params_config.get("screen_height", screen_height_derived)

    screen_corners = np.array([
        [-screen_width/2, -screen_height/2, screen_distance],
        [screen_width/2, -screen_height/2, screen_distance],
        [screen_width/2, screen_height/2, screen_distance],
        [-screen_width/2, screen_height/2, screen_distance],
    ])
    screen_verts = [screen_corners]
    screen_collection = Poly3DCollection(screen_verts, alpha=0.2, color='cyan', label='Camera Screen')
    ax.add_collection3d(screen_collection)
    ax.text(0, 0, screen_distance, " Screen", color='black', fontsize=8)


    # トップダウングリッドを表現 (coordinate_systems.py を参考に値を調整)
    grid_y_level = camera_params_config.get("top_down_grid_y_level", 0.5)  # Y軸の値（床の高さなど）
    grid_size_x = camera_params_config.get("top_down_grid_size_x", 2.0) # X方向のサイズ
    grid_size_z = camera_params_config.get("top_down_grid_size_z", 3.0) # Z方向のサイズ
    grid_z_offset = camera_params_config.get("top_down_grid_z_offset", 0.5) # Z方向のオフセット

    grid_corners = np.array([
        [-grid_size_x/2, grid_y_level, grid_z_offset],
        [grid_size_x/2, grid_y_level, grid_z_offset],
        [grid_size_x/2, grid_y_level, grid_z_offset + grid_size_z],
        [-grid_size_x/2, grid_y_level, grid_z_offset + grid_size_z],
    ])
    grid_verts = [grid_corners]
    grid_collection = Poly3DCollection(grid_verts, alpha=0.2, color='lightgreen', label='Top-down Grid')
    ax.add_collection3d(grid_collection)
    ax.text(0, grid_y_level, grid_z_offset + grid_size_z/2, " Grid", color='black', fontsize=8)

    if points is not None and points.size > 0:
        # ポイントクラウドのY軸はカメラ座標系に従い下向きが正
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis', alpha=0.7)
    else:
        print("No points to visualize or points array is invalid.")

    # 軸ラベルとタイトル
    ax.set_xlabel("X (右)")
    ax.set_ylabel("Y (下)")
    ax.set_zlabel("Z (前)")
    ax.set_title("CSV深度データからの3Dポイントクラウドと座標系")

    # 視点調整 (Y軸下向きを考慮)
    ax.view_init(elev=-24, azim=-170, roll=85) # 前から見る
    # ax.view_init(elev=20, azim=-75) # 少し上から斜め
    # ax.view_init(elev=0, azim=-90) # 真横から (X-Z平面)
    # ax.view_init(elev=-90, azim=-90) # 真下から (X-Z平面、トップダウンに近いがY軸反転)

    # 軸の範囲設定 (ポイントクラウドや平面に合わせて調整)
    # ax.set_xlim([-1.5, 1.5])
    # ax.set_ylim([-0.5, 1.5]) # Y軸下向きなので、-0.5 (上) から 1.5 (下)
    # ax.set_zlim([0, 3.5])
    # ax.set_box_aspect([3, 2, 3.5]) # X, Y, Z の軸スケール比

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSVファイルから深度データを読み込み3Dで可視化します。")
    parser.add_argument("csv_file", type=str, help="深度データCSVファイルのパス。")
    parser.add_argument("--config", type=str, default="config.json", help="設定ファイルのパス。")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # カメラと深度設定の取得
    # camera_config = config.get("camera", {}) # あまり使わないかも
    depth_config = config.get("depth", {})

    fx = depth_config.get("fx")
    fy = depth_config.get("fy")
    cx = depth_config.get("cx")
    cy = depth_config.get("cy")
    scaling_factor = depth_config.get("scaling_factor") # convert_to_absolute_depthで使う
    
    # config.json内のwidth, heightはモデルの入力解像度を指すことが多い
    # ポイントクラウド生成時の original_width, original_height は、fx,fy,cx,cyがどの解像度に対応するかの情報
    original_width = depth_config.get("width") 
    original_height = depth_config.get("height")

    if not all([fx, fy, cx, cy, scaling_factor, original_width, original_height]):
        print("エラー: 設定ファイルに必要なカメラパラメータ (fx, fy, cx, cy, scaling_factor, width, height) が見つかりません。")
        sys.exit(1)

    depth_data_raw = load_depth_data_from_csv(args.csv_file)
    
    # scaling_factor がリストや数値でない場合の対応を追加
    if isinstance(scaling_factor, list):
        # リストの場合、適切な値を選択するか、エラー処理
        # ここでは仮に最初の値を使用するか、あるいは特定のロジックが必要
        if len(scaling_factor) > 0:
            s_factor = scaling_factor[0] 
            print(f"Warning: scaling_factor is a list, using the first element: {s_factor}")
        else:
            print("Error: scaling_factor is an empty list.")
            sys.exit(1)
    elif isinstance(scaling_factor, (int, float)):
        s_factor = scaling_factor
    else:
        print(f"Error: scaling_factor is of an unsupported type: {type(scaling_factor)}")
        sys.exit(1)

    depth_absolute = convert_to_absolute_depth(depth_data_raw, s_factor)

    # depth_to_point_cloud に渡すカメラパラメータを辞書にまとめる
    # スケーリングされた内部パラメータを計算
    # fx, fy, cx, cy は original_width, original_height における値
    # depth_data_raw.shape[1] (cols), depth_data_raw.shape[0] (rows) はCSVの解像度
    
    # CSVの解像度とconfigの解像度が異なる場合、焦点距離と光軸中心をスケーリングする必要がある
    grid_rows, grid_cols = depth_data_raw.shape
    
    fx_scaled = fx * (grid_cols / original_width) if original_width > 0 else fx
    fy_scaled = fy * (grid_rows / original_height) if original_height > 0 else fy
    cx_scaled = cx * (grid_cols / original_width) if original_width > 0 else cx
    cy_scaled = cy * (grid_rows / original_height) if original_height > 0 else cy

    # camera_params_for_pc は直接使わなくなるか、他の目的で使用
    # camera_params_for_pc = {
    #     "fx": fx_scaled,
    #     "fy": fy_scaled,
    #     "cx": cx_scaled,
    #     "cy": cy_scaled,
    #     "is_grid_data": True 
    # }
    
    # visualize_point_cloud_3d に渡す設定用辞書にも追加しておく
    camera_params_config_vis = {
        "fx_scaled": fx_scaled, "fy_scaled": fy_scaled, "cx_scaled": cx_scaled, "cy_scaled": cy_scaled,
        "screen_distance": 0.3, # 例: スクリーンのZ位置
        "top_down_grid_y_level": 0.5, # 例: グリッドのYレベル (床面など)
        # 他の平面表示用パラメータもここに追加可能
    }

    points = depth_to_point_cloud(
        depth_absolute,
        fx=fx_scaled,
        fy=fy_scaled,
        cx=cx_scaled,
        cy=cy_scaled,
        is_grid_data=True # CSVデータはグリッドデータとして扱う
    )
    
    visualize_point_cloud_3d(points, camera_params_config_vis, depth_data_raw.shape) # depth_grid_shape を渡す
