"""
create_top_down_occupancy_grid関数の修正結果を確認するためのファイル
"""
import numpy as np
import matplotlib.pyplot as plt
from depth_processor.point_cloud import depth_to_point_cloud, create_top_down_occupancy_grid
from depth_processor.depth_model import convert_to_absolute_depth

def test_fixed_occupancy_grid():
    """修正された関数の挙動を検証する"""
    # 単純なテストケース
    points = np.array([
        [-0.69230769,  0.69230769,  1.38461538],
        [ 0.5,         0.5,         1.0 ],
        [ 0.92307692,  0.30769231,  0.61538462],
        [ 0.75,        0.75,        0.5 ]
    ])
    
    # パラメータ設定
    grid_params = {
        "x_range": (-1.0, 1.0),
        "z_range": (0.0, 2.0),
        "resolution": 0.5,
        "y_min": 0.0,
        "y_max": 1.0,
        "use_adaptive_thresholds": False
    }
    
    print("テスト点のデータ:")
    print(points)
    print("\n占有グリッドの生成...")
    
    # 占有グリッドを作成
    occupancy_grid = create_top_down_occupancy_grid(points, grid_params)
    
    # 結果の確認
    print("\n占有グリッド:")
    print(occupancy_grid)
    
    # 投影点を計算
    projected_points = points[(points[:, 1] >= grid_params["y_min"]) & 
                             (points[:, 1] <= grid_params["y_max"])]
    
    # 一貫性を確認
    correct = True
    for point in projected_points:
        x, z = point[0], point[2]
        grid_x = int((x - grid_params["x_range"][0]) / grid_params["resolution"])
        grid_z = int((z - grid_params["z_range"][0]) / grid_params["resolution"])
        
        if 0 <= grid_x < occupancy_grid.shape[1] and 0 <= grid_z < occupancy_grid.shape[0]:
            if occupancy_grid[grid_z, grid_x] <= 0:
                correct = False
                print(f"エラー: 点 {point} がグリッド上に反映されていません")
    
    if correct:
        print("\nテスト成功: すべての投影点が占有グリッドに反映されました")
    
    # 可視化
    plt.figure(figsize=(8, 6))
    plt.imshow(occupancy_grid, cmap='viridis', origin='lower')
    plt.title('修正された占有グリッド')
    plt.colorbar(label='占有状態')
    
    # 投影点をプロット
    for point in projected_points:
        x, z = point[0], point[2]
        grid_x = int((x - grid_params["x_range"][0]) / grid_params["resolution"])
        grid_z = int((z - grid_params["z_range"][0]) / grid_params["resolution"])
        plt.plot(grid_x, grid_z, 'rx', markersize=12, markeredgewidth=2)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("occupancy_grid_test.png")
    plt.close()
    
    print("結果を 'occupancy_grid_test.png' として保存しました")
    
    return occupancy_grid, projected_points

if __name__ == "__main__":
    grid, points = test_fixed_occupancy_grid()
