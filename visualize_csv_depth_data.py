\
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

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

def visualize_point_cloud_3d(points, camera_params_config):
    """3Dポイントクラウドとカメラ座標系をMatplotlibで可視化する"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Matplotlibの日本語フォント設定 (coordinate_systems.pyと同様)
    # matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP' # Or your preferred font

    # カメラ座標系の軸を描画
    axis_length = 0.5  # ポイントクラウドに焦点を合わせるため少し小さく
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='Camera X (右)')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Camera Y (下)') # Y軸は下向き
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Camera Z (前)')

    if points is not None and points.size > 0:
        # Y軸の値を反転して表示（カメラ座標系Y軸下向きを、一般的な3Dプロットの上向きに合わせる場合）
        # ただし、ここではカメラ座標系に従い、Y下向きのままプロットする。
        # 必要であれば points[:, 1] * -1 で反転
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis', alpha=0.5)
        print(f"Visualizing {points.shape[0]} points.")
    else:
        print("No points to visualize or points array is invalid.")

    ax.set_xlabel('X (メートル)')
    ax.set_ylabel('Y (メートル)')
    ax.set_zlabel('Z (メートル)')
    ax.set_title('CSVからの3Dポイントクラウド')

    # 視点や範囲、アスペクト比はcoordinate_systems.pyを参考に調整可能
    # ax.set_xlim([-3, 3])
    # ax.set_ylim([-3, 3]) # Y軸の範囲も考慮
    # ax.set_zlim([0, 6])
    # ax.view_init(elev=20, azim=-75) # 例: 少し上からの視点
    ax.view_init(elev=-150, azim=-90) # Y軸下向きを考慮した視点調整の例 (前から見る)
    # ax.view_init(elev=0, azim=-90) # 真横からの視点

    # Z軸を上向きとして表示したい場合は、座標変換とview_initの調整が必要
    # ax.set_box_aspect([1,1,1]) # アスペクト比を均等に

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

    # CSVから深度グリッドを読み込む (相対深度を想定)
    relative_depth_grid = load_depth_data_from_csv(args.csv_file)

    if relative_depth_grid is None:
        sys.exit(1)

    # 相対深度グリッドを絶対深度(メートル単位)に変換
    # is_compressed_grid=True はCSVが圧縮グリッドであることを示す
    absolute_depth_grid = convert_to_absolute_depth(
        relative_depth_grid,
        scaling_factor=scaling_factor,
        is_compressed_grid=True 
    )
    print(f"絶対深度グリッドに変換完了, shape: {absolute_depth_grid.shape}")

    grid_rows, grid_cols = absolute_depth_grid.shape

    # ポイントクラウド生成のためのカメラパラメータ調整
    # fx, fy はピクセル単位の焦点距離。cx, cy はピクセル単位の光軸中心。
    # これらが original_width, original_height の解像度に対するものであると仮定。
    # depth_to_point_cloud は is_grid_data=True の場合、
    # fx, fy を「グリッドセル単位の焦点距離」として、
    # cx, cy を「グリッドセル単位の光軸中心」として期待する。

    fx_for_grid = fx * (grid_cols / original_width)
    fy_for_grid = fy * (grid_rows / original_height)
    
    # cx, cy もグリッドの解像度に合わせてスケーリング
    cx_for_grid = cx * (grid_cols / original_width)
    cy_for_grid = cy * (grid_rows / original_height)
    
    print(f"ポイントクラウド生成用パラメータ: fx_grid={fx_for_grid:.2f}, fy_grid={fy_for_grid:.2f}, cx_grid={cx_for_grid:.2f}, cy_grid={cy_for_grid:.2f}")

    points = depth_to_point_cloud(
        absolute_depth_grid,
        fx=fx_for_grid,
        fy=fy_for_grid,
        cx=cx_for_grid, 
        cy=cy_for_grid,
        is_grid_data=True,
        grid_rows=grid_rows,
        grid_cols=grid_cols
        # original_width, original_height は直接使われないが、
        # fx_for_grid等の計算の前提となっている解像度情報として重要
    )

    if points is not None and points.ndim == 2 and points.shape[1] == 3:
        visualize_point_cloud_3d(points, depth_config)
    else:
        print("ポイントクラウドの生成に失敗したか、ポイントがありません。")
