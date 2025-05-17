import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys
import traceback  # デバッグ用に追加

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from depth_processor.point_cloud import depth_to_point_cloud, create_top_down_occupancy_grid
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

def visualize_point_cloud_3d(points, camera_params_config, depth_grid_shape):
    """3Dポイントクラウドとカメラ座標系、各種平面をMatplotlibで可視化する"""
    # グラフサイズを調整（画面に収まりやすいサイズに）
    fig = plt.figure(figsize=(12, 8))
    
    # 単一の3Dグラフを使用
    ax_3d = fig.add_subplot(111, projection='3d')
    
    # ==== 3Dビューの設定 === =
    # カメラ座標系の軸を描画
    axis_length = 0.5
    ax_3d.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='Camera X (右)')
    ax_3d.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Camera Y (下)')
    ax_3d.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Camera Z (前)')

    # カメラスクリーンを表現
    screen_distance = camera_params_config.get("screen_distance", 0.3)
    screen_width = 0.6
    screen_height = 0.4

    screen_corners = np.array([
        [-screen_width/2, -screen_height/2, screen_distance],
        [screen_width/2, -screen_height/2, screen_distance],
        [screen_width/2, screen_height/2, screen_distance],
        [-screen_width/2, screen_height/2, screen_distance],
    ])
    screen_verts = [screen_corners]
    screen_collection = Poly3DCollection(screen_verts, alpha=0.2, color='cyan', label='Camera Screen')
    ax_3d.add_collection3d(screen_collection)
    ax_3d.text(0, 0, screen_distance, " Screen", color='black', fontsize=8)

    # トップダウングリッドを表現
    grid_y_level = camera_params_config.get("top_down_grid_y_level", 0.5)
    grid_size_x = camera_params_config.get("top_down_grid_size_x", 2.0)
    grid_size_z = camera_params_config.get("top_down_grid_size_z", 3.0)
    grid_z_offset = camera_params_config.get("top_down_grid_z_offset", 0.5)

    grid_corners = np.array([
        [-grid_size_x/2, grid_y_level, grid_z_offset],
        [grid_size_x/2, grid_y_level, grid_z_offset],
        [grid_size_x/2, grid_y_level, grid_z_offset + grid_size_z],
        [-grid_size_x/2, grid_y_level, grid_z_offset + grid_size_z],
    ])
    grid_verts = [grid_corners]
    grid_collection = Poly3DCollection(grid_verts, alpha=0.2, color='lightgreen', label='Top-down Grid')
    ax_3d.add_collection3d(grid_collection)
    ax_3d.text(0, grid_y_level, grid_z_offset + grid_size_z/2, " Grid", color='black', fontsize=8)

    if points is not None and points.size > 0:
        # ポイントのサイズを大きくする（3Dプロット用）
        ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c=points[:, 2], cmap='viridis', alpha=0.7, label='Point Cloud')
    else:
        print("No points to visualize or points array is invalid.")
        
    # ==== トップダウン投影点の表示を3Dグラフに追加 === =
    if points is not None and points.size > 0:
        # 点群のY座標（高さ）の範囲を自動的に計算
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        print(f"点群の高さ範囲: {y_min:.2f}m 〜 {y_max:.2f}m")
        
        # トップダウンビューのパラメータ - 高さ範囲を調整
        # 自動的にY座標の範囲から適切な範囲を計算
        min_height = max(y_min, -4.0)   # 下限（床面から）を設定、極端な外れ値を除外
        max_height = min(y_max, 1.5)    # 上限を設定、天井などの外れ値を除外
        
        print(f"トップダウン投影で使用する高さ範囲: {min_height:.2f}m 〜 {max_height:.2f}m")
        
        # トップダウンビューへの点の投影（XZ平面）
        valid_points = points[(points[:, 1] >= min_height) & (points[:, 1] <= max_height)]
        print(f"有効な点の数: {len(valid_points)}/{len(points)} ({len(valid_points)/len(points)*100:.1f}%)")
        
        if valid_points.size > 0:
            # 投影点を3Dグラフにプロット（グリッド平面上）
            ax_3d.scatter(
                valid_points[:, 0],                    # X座標はそのまま 
                np.full(valid_points.shape[0], grid_y_level),  # Y座標はグリッド平面のY値に
                valid_points[:, 2],                    # Z座標はそのまま
                s=20,                                  # ポイントサイズ
                c='red', 
                marker='x',                            # マーカーをXに
                alpha=0.8,                             # 透明度
                label='Projected Points'               # ラベル
            )
            
            # サンプル数を調整（データ量が多い場合にはさらに少なく）
            sample_count = min(30, len(valid_points))
            sample_indices = np.linspace(0, len(valid_points)-1, sample_count).astype(int)
            
            # 元のポイントから投影点への線を描画（選択したサンプルのみ）
            for i in sample_indices:
                ax_3d.plot(
                    [valid_points[i, 0], valid_points[i, 0]],           # X座標（変化なし）
                    [valid_points[i, 1], grid_y_level],                  # Y座標（元の高さ→グリッド平面）
                    [valid_points[i, 2], valid_points[i, 2]],           # Z座標（変化なし）
                    'k:', alpha=0.3                                      # 点線、やや透明
                )
            
            # 占有グリッドの計算と表示
            # サンプル数やパフォーマンスに応じて解像度を調整
            if len(valid_points) > 500:
                resolution = 0.2  # 大量のポイントがある場合は粗い解像度
            elif len(valid_points) > 200:
                resolution = 0.15 # 中程度のポイント数
            else:
                resolution = 0.1  # 少数のポイントの場合は詳細な解像度

            # 実際の点群の範囲を使用してグリッドパラメータを設定
            if valid_points.size > 0:
                # 実際のデータから範囲を計算
                x_min, x_max = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
                z_min, z_max = np.min(valid_points[:, 2]), np.max(valid_points[:, 2])
                
                # 少し余裕を持たせる
                x_padding = max(0.2, (x_max - x_min) * 0.1)
                z_padding = max(0.2, (z_max - z_min) * 0.1)
                
                grid_params = {
                    "x_range": (x_min - x_padding, x_max + x_padding),
                    "z_range": (z_min - z_padding, z_max + z_padding),
                    "resolution": resolution,
                    "y_min": min_height,
                    "y_max": max_height,
                    "use_adaptive_thresholds": False,  # 安定した結果のため適応閾値を無効化
                    "obstacle_threshold": 0.1  # 障害物として検出する高さの閾値
                }
                
                # グリッドの範囲をトップダウンプレーンにも反映
                # グリッド平面も更新
                updated_grid_corners = np.array([
                    [grid_params["x_range"][0], grid_y_level, grid_params["z_range"][0]],
                    [grid_params["x_range"][1], grid_y_level, grid_params["z_range"][0]],
                    [grid_params["x_range"][1], grid_y_level, grid_params["z_range"][1]],
                    [grid_params["x_range"][0], grid_y_level, grid_params["z_range"][1]],
                ])
                
                # 既存のグリッド平面を削除して新しいものを表示
                for collection in ax_3d.collections:
                    try:
                        if (hasattr(collection, '_edgecolors') and 
                            hasattr(collection, '_facecolors') and 
                            collection._facecolors.size > 0 and
                            np.array_equal(collection._facecolors[0][:3], matplotlib.colors.to_rgb('lightgreen'))):
                            collection.remove()
                    except (IndexError, AttributeError):
                        continue
                
                updated_grid_verts = [updated_grid_corners]
                updated_grid_collection = Poly3DCollection(updated_grid_verts, alpha=0.2, color='lightgreen', label='Updated Grid')
                ax_3d.add_collection3d(updated_grid_collection)
                
                # 占有グリッドの処理
                try:
                    print("占有グリッドの生成を開始...")
                    print(f"グリッドパラメータ: {grid_params}")
                    
                    # 占有グリッドの作成
                    occupancy_grid = create_top_down_occupancy_grid(valid_points, grid_params)
                    
                    # グリッドの各セルの中心座標を計算
                    grid_height, grid_width = occupancy_grid.shape
                    x_coords = np.linspace(grid_params["x_range"][0] + resolution/2, grid_params["x_range"][1] - resolution/2, grid_width)
                    z_coords = np.linspace(grid_params["z_range"][0] + resolution/2, grid_params["z_range"][1] - resolution/2, grid_height)
                    
                    print(f"占有グリッドの形状: {occupancy_grid.shape}, 占有セル数: {np.sum(occupancy_grid > 0)}")
                    print(f"占有グリッドの値: 最小={np.min(occupancy_grid)}, 最大={np.max(occupancy_grid)}")
                    
                    # デバッグ: 占有グリッドの中身を表示
                    if np.sum(occupancy_grid > 0) == 0:
                        print("警告: 占有グリッドに値が入っていません")
                    else:
                        # 一定数の占有セルの値を表示
                        occupied_cells = np.where(occupancy_grid > 0)
                        if len(occupied_cells[0]) > 0:
                            print("占有セルのサンプル:")
                            for i in range(min(5, len(occupied_cells[0]))):
                                row, col = occupied_cells[0][i], occupied_cells[1][i]
                                print(f"  セル[{row},{col}] = {occupancy_grid[row, col]}, 座標 = ({z_coords[row]:.2f}, {x_coords[col]:.2f})")
                    
                    # 占有セルの表示
                    occupied_count = 0
                    for i in range(grid_height):
                        for j in range(grid_width):
                            cell_value = occupancy_grid[i, j]
                            if cell_value > 0:  # 占有されているセル
                                occupied_count += 1
                                x_center = x_coords[j]
                                z_center = z_coords[i]
                                
                                # セルを色付きの立方体として表示（高さもある）
                                cell_size_y = 0.1  # セルの高さ
                                cell_corners = np.array([
                                    [x_center - resolution/2, grid_y_level, z_center - resolution/2],
                                    [x_center + resolution/2, grid_y_level, z_center - resolution/2],
                                    [x_center + resolution/2, grid_y_level, z_center + resolution/2],
                                    [x_center - resolution/2, grid_y_level, z_center + resolution/2],
                                    [x_center - resolution/2, grid_y_level + cell_size_y, z_center - resolution/2],
                                    [x_center + resolution/2, grid_y_level + cell_size_y, z_center - resolution/2],
                                    [x_center + resolution/2, grid_y_level + cell_size_y, z_center + resolution/2],
                                    [x_center - resolution/2, grid_y_level + cell_size_y, z_center + resolution/2]
                                ])
                                
                                # ここは立方体を各面ごとに描画
                                faces = [
                                    [cell_corners[0], cell_corners[1], cell_corners[2], cell_corners[3]],  # 底面
                                    [cell_corners[4], cell_corners[5], cell_corners[6], cell_corners[7]],  # 上面
                                    [cell_corners[0], cell_corners[1], cell_corners[5], cell_corners[4]],  # 前面
                                    [cell_corners[1], cell_corners[2], cell_corners[6], cell_corners[5]],  # 右面
                                    [cell_corners[2], cell_corners[3], cell_corners[7], cell_corners[6]],  # 後面
                                    [cell_corners[3], cell_corners[0], cell_corners[4], cell_corners[7]]   # 左面
                                ]
                                
                                # 占有度に応じた色を使用 (暖色系で、より目立つように)
                                color_value = min(1.0, cell_value / np.max(occupancy_grid) if np.max(occupancy_grid) > 0 else 0)
                                cell_color = plt.cm.hot(color_value)  # 暖色系カラーマップ
                                cell_collection = Poly3DCollection(faces, alpha=0.7, facecolor=cell_color, edgecolor='black')
                                ax_3d.add_collection3d(cell_collection)
                    
                    print(f"グラフに表示した占有セル数: {occupied_count}")
                    
                except Exception as e:
                    print(f"占有グリッド作成エラー: {e}")
                    traceback.print_exc()  # 詳細なエラー情報を表示
    
    # 軸ラベルとタイトル（3D）
    ax_3d.set_xlabel("X (右)")
    ax_3d.set_ylabel("Y (下)")
    ax_3d.set_zlabel("Z (前)")
    ax_3d.set_title("3Dポイントクラウドとトップダウン投影")

    # 視点調整（見やすい角度に設定）
    ax_3d.view_init(elev=30, azim=-135)
    
    # 軸の範囲を調整（全体を表示するために少し広めに）
    if points is not None and points.size > 0:
        # データに基づいて軸範囲を設定
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        
        # 余裕を持たせる
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        z_padding = (z_max - z_min) * 0.1
        
        ax_3d.set_xlim([x_min - x_padding, x_max + x_padding])
        ax_3d.set_ylim([y_min - y_padding, y_max + y_padding])
        ax_3d.set_zlim([z_min - z_padding, z_max + z_padding])
    else:
        ax_3d.set_xlim([-grid_size_x/2-0.3, grid_size_x/2+0.3])
        ax_3d.set_ylim([-0.3, 1.3])
        ax_3d.set_zlim([0, grid_size_z+0.3])
    
    # レジェンドと全体の調整
    ax_3d.legend(loc='upper right')
    
    # グリッド表示
    ax_3d.grid(True)
    
    # マウスホイールによるズーム対応
    def on_scroll(event):
        # ホイールの方向に応じてズーム倍率を決定
        zoom_factor = 0.9 if event.button == 'down' else 1.1
        
        # 現在の軸の範囲を取得
        x_min, x_max = ax_3d.get_xlim()
        y_min, y_max = ax_3d.get_ylim()
        z_min, z_max = ax_3d.get_zlim()
        
        # 新しい範囲を計算（中心点を基準にズーム）
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        x_range = (x_max - x_min) / 2 * zoom_factor
        y_range = (y_max - y_min) / 2 * zoom_factor
        z_range = (z_max - z_min) / 2 * zoom_factor
        
        # 新しい範囲を設定
        ax_3d.set_xlim([x_center - x_range, x_center + x_range])
        ax_3d.set_ylim([y_center - y_range, y_center + y_range])
        ax_3d.set_zlim([z_center - z_range, z_center + z_range])
        
        fig.canvas.draw_idle()
    
    # ズーム用のスクロールイベントを接続
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # 座標変換の説明テキスト
    fig.text(0.02, 0.02, '''座標系: 
- カメラ座標系: X(右), Y(下), Z(前)
- 投影点: カメラ座標系の点をY=固定値の平面に投影
- 操作: マウスホイールでズーム''', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSVファイルから深度データを読み込み3Dで可視化します。")
    parser.add_argument("csv_file", type=str, help="深度データCSVファイルのパス。")
    parser.add_argument("--config", type=str, default="config.json", help="設定ファイルのパス。")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # カメラと深度設定の取得
    depth_config = config.get("depth", {})

    fx = depth_config.get("fx")
    fy = depth_config.get("fy")
    cx = depth_config.get("cx")
    cy = depth_config.get("cy")
    scaling_factor = depth_config.get("scaling_factor")
    original_width = depth_config.get("width") 
    original_height = depth_config.get("height")

    if not all([fx, fy, cx, cy, scaling_factor, original_width, original_height]):
        print("エラー: 設定ファイルに必要なカメラパラメータが見つかりません。")
        sys.exit(1)

    depth_data_raw = load_depth_data_from_csv(args.csv_file)
    
    # scaling_factorの処理
    if isinstance(scaling_factor, list):
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
    
    # CSVの解像度とconfigの解像度のスケール調整
    grid_rows, grid_cols = depth_data_raw.shape
    
    fx_scaled = fx * (grid_cols / original_width) if original_width > 0 else fx
    fy_scaled = fy * (grid_rows / original_height) if original_height > 0 else fy
    cx_scaled = cx * (grid_cols / original_width) if original_width > 0 else cx
    cy_scaled = cy * (grid_rows / original_height) if original_height > 0 else cy
    
    camera_params_config_vis = {
        "fx_scaled": fx_scaled, 
        "fy_scaled": fy_scaled, 
        "cx_scaled": cx_scaled, 
        "cy_scaled": cy_scaled,
        "screen_distance": 0.3,
        "top_down_grid_y_level": 0.5,
    }

    points = depth_to_point_cloud(
        depth_absolute,
        fx=fx_scaled,
        fy=fy_scaled,
        cx=cx_scaled,
        cy=cy_scaled,
        is_grid_data=True
    )
    
    # 単一の関数呼び出しで全ての処理を行う
    visualize_point_cloud_3d(points, camera_params_config_vis, depth_data_raw.shape)

    # ここ以降のすべてのコードを削除
    # 占有グリッドの計算と表示は関数内で完結させる

