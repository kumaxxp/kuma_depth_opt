"""
create_top_down_occupancy_grid関数の問題を調査するためのスクリプト
"""
import numpy as np
import sys
import os
import logging
from depth_processor.point_cloud import depth_to_point_cloud, create_top_down_occupancy_grid
from depth_processor.depth_model import convert_to_absolute_depth

# ロギングの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("occupancy_grid_investigation")

def investigate_function():
    """create_top_down_occupancy_grid関数の問題を特定する"""
    logger.info("Starting investigation of create_top_down_occupancy_grid...")
    
    # テスト用の簡単なポイントクラウドを作成
    points = np.array([
        [-0.69230769,  0.69230769,  1.38461538],  # これは元のテストで問題になったポイント
        [ 0.5,         0.5,         1.0 ],
        [ 0.92307692,  0.30769231,  0.61538462],
        [ 0.75,        0.75,        0.5 ]
    ])
    
    # トップダウン投影のパラメータ
    base_params = {
        "x_range": (-1.0, 1.0),
        "z_range": (0.0, 2.0),
        "resolution": 0.5,
        "y_min": 0.0,
        "y_max": 1.0
    }
    
    # 関数のソースを確認（可能であれば）
    try:
        import inspect
        source = inspect.getsource(create_top_down_occupancy_grid)
        logger.info(f"Function source:\n{source}")
    except Exception as e:
        logger.warning(f"Could not retrieve function source: {e}")
    
    # さまざまなパラメータでテスト
    test_cases = [
        # ベースケース
        {"name": "Base case", "params": base_params.copy()},
        
        # 高さ閾値を調整
        {"name": "Lower y_min", "params": {**base_params, "y_min": -1.0}},
        {"name": "Higher y_max", "params": {**base_params, "y_max": 2.0}},
        
        # obstacle_thresholdを追加
        {"name": "With obstacle_threshold", "params": {**base_params, "obstacle_threshold": 0.1}},
        
        # 適応閾値を無効化
        {"name": "Disable adaptive thresholds", "params": {**base_params, "use_adaptive_thresholds": False}},
        
        # 範囲を調整
        {"name": "Extended x_range", "params": {**base_params, "x_range": (-2.0, 2.0)}},
        {"name": "Extended z_range", "params": {**base_params, "z_range": (-1.0, 3.0)}},
    ]
    
    # それぞれのケースをテスト
    for case in test_cases:
        logger.info(f"Testing: {case['name']}")
        params = case["params"]
        
        try:
            grid = create_top_down_occupancy_grid(points, params)
            logger.info(f"Grid shape: {grid.shape}")
            logger.info(f"Grid sum: {np.sum(grid)}")
            logger.info(f"Grid:\n{grid}")
            
            # 投影点とグリッドの一貫性をチェック
            check_consistency(points, grid, params)
            
        except Exception as e:
            logger.error(f"Error during {case['name']}: {e}")
    
    # 独自の実装と比較
    custom_grid = create_custom_occupancy_grid(points, base_params)
    logger.info("Custom implementation grid:")
    logger.info(f"Grid shape: {custom_grid.shape}")
    logger.info(f"Grid sum: {np.sum(custom_grid)}")
    logger.info(f"Grid:\n{custom_grid}")
    
    return "Investigation completed"

def check_consistency(points, grid, params):
    """投影点とグリッドの一貫性をチェック"""
    projected_points = points[(points[:, 1] >= params.get("y_min", 0.0)) & 
                             (points[:, 1] <= params.get("y_max", 1.0))]
    
    if len(projected_points) == 0:
        logger.warning("No points within height range")
        return
    
    logger.info(f"Projected points count: {len(projected_points)}")
    
    for point in projected_points:
        x, z = point[0], point[2]
        grid_x = int((x - params["x_range"][0]) / params["resolution"])
        grid_z = int((z - params["z_range"][0]) / params["resolution"])
        
        if 0 <= grid_x < grid.shape[1] and 0 <= grid_z < grid.shape[0]:
            cell_value = grid[grid_z, grid_x]
            logger.info(f"Point {point} -> Cell ({grid_z}, {grid_x}) = {cell_value}")
            if cell_value <= 0:
                logger.warning(f"Point {point} not reflected in grid")
        else:
            logger.warning(f"Point {point} outside grid bounds: ({grid_z}, {grid_x})")

def create_custom_occupancy_grid(points, params):
    """カスタムの占有グリッド生成関数"""
    x_min, x_max = params["x_range"]
    z_min, z_max = params["z_range"]
    resolution = params["resolution"]
    y_min = params.get("y_min", 0.0)
    y_max = params.get("y_max", 1.0)
    
    # グリッドサイズを計算
    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((z_max - z_min) / resolution)
    
    # グリッドを初期化
    grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    # 高さで点をフィルタリング
    projected_points = points[(points[:, 1] >= y_min) & (points[:, 1] <= y_max)]
    
    # 投影点をグリッドに反映
    for point in projected_points:
        x, z = point[0], point[2]
        grid_x = int((x - x_min) / resolution)
        grid_z = int((z - z_min) / resolution)
        
        if 0 <= grid_x < grid_width and 0 <= grid_z < grid_height:
            grid[grid_z, grid_x] = 1.0
    
    return grid

if __name__ == "__main__":
    result = investigate_function()
    print(result)
