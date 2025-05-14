"""
深度マップから点群を生成し、トップビュー（天頂視点）表示を行う機能を提供します。
"""

import numpy as np
import cv2
import logging

# ロガーの取得と設定
logger = logging.getLogger("kuma_depth_opt.point_cloud")
logger.setLevel(logging.DEBUG)
# ハンドラが設定されていなければ、標準出力へのハンドラを追加
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.info("Logger for 'kuma_depth_opt.point_cloud' INITIALIZED.")

# デフォルトパラメータ
GRID_RESOLUTION = 0.06  # メートル/セル
GRID_WIDTH = 100        # 横方向のセル数
GRID_HEIGHT = 100       # 縦方向のセル数
HEIGHT_THRESHOLD = 0.3  # 通行可能と判定する高さの閾値（メートル）
MAX_DEPTH = 6.0         # 最大深度（メートル）

def depth_to_point_cloud(depth_data, fx, fy, cx, cy,
                         original_height=None, original_width=None,
                         is_grid_data=False, grid_rows=None, grid_cols=None):
    """
    深度データから3D点群を生成します。
    高解像度深度マップと圧縮グリッドデータの両方に対応します。
    
    Args:
        depth_data (numpy.ndarray): 深度データ。フル解像度マップまたは圧縮グリッド。
        fx (float): カメラの水平方向の焦点距離。
        fy (float): カメラの垂直方向の焦点距離。
        cx (float): カメラの水平方向の光学中心。
        cy (float): カメラの垂直方向の光学中心。
        original_height (int, optional): グリッドデータの場合の元の深度マップの高さ。
        original_width (int, optional): グリッドデータの場合の元の深度マップの幅。
        is_grid_data (bool): Trueの場合、depth_dataを圧縮グリッドとして処理。
        grid_rows (int, optional): グリッドデータの場合の行数。
        grid_cols (int, optional): グリッドデータの場合の列数。

    Returns:
        numpy.ndarray: (N, 3)形状の点群データ。各点は[x, y, z]。
                       無効な点や点がない場合は空の配列。
    """
    try:
        # デバッグ出力
        logger.debug(f"[PointCloud] Input depth_data shape: {depth_data.shape}, range: {np.min(depth_data):.4f} to {np.max(depth_data):.4f}")
        logger.debug(f"[PointCloud] Camera params: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        if is_grid_data:
            logger.debug(f"[PointCloud] Grid mode with rows={grid_rows}, cols={grid_cols}, original size={original_height}x{original_width}")
        else:
            logger.debug("[PointCloud] Full resolution mode")
        
        # 入力チェック
        if depth_data is None or depth_data.size == 0:
            logger.warning("[PointCloud] Error: Empty depth data")
            return np.empty((0, 3), dtype=np.float32)
        
        if is_grid_data:
            # グリッドデータのパラメータチェック
            if grid_rows is None or grid_cols is None:
                # 実際のデータ形状から推定
                grid_rows, grid_cols = depth_data.shape[:2]
                logger.debug(f"[PointCloud] Using actual grid dimensions: {grid_rows}x{grid_cols}")
            
            # 圧縮データから直接点群生成（ベクトル化処理で高速化）
            # グリッドのインデックスを作成
            r_indices, c_indices = np.indices((grid_rows, grid_cols))
            
            # 有効な深度値の判定
            valid_mask = depth_data > 0.01
            valid_depth = depth_data[valid_mask]
            
            if valid_depth.size == 0:
                logger.warning("[PointCloud] No valid depth values in grid data")
                return np.empty((0, 3), dtype=np.float32)
                
            valid_r = r_indices[valid_mask]
            valid_c = c_indices[valid_mask]
            
            # グリッドセルに対応する中心ピクセル座標を計算
            u_centers = (valid_c + 0.5)  # グリッド上での中心座標
            v_centers = (valid_r + 0.5)
            
            # 3D座標を一括計算
            # 注意: cx, cyはグリッドの中心点を使用
            x_values = (u_centers - cx) * valid_depth / fx
            y_values = (v_centers - cy) * valid_depth / fy
            z_values = valid_depth
            
            # 無効な値をフィルタリング
            valid_idx = ~(np.isnan(x_values) | np.isnan(y_values) | np.isnan(z_values) |
                          np.isinf(x_values) | np.isinf(y_values) | np.isinf(z_values))
            valid_idx &= (np.abs(x_values) < 10) & (np.abs(y_values) < 10) & (z_values < 20) & (z_values > 0)
            
            if np.sum(valid_idx) == 0:
                logger.warning("[PointCloud] No valid points after filtering")
                return np.empty((0, 3), dtype=np.float32)
                
            x_values = x_values[valid_idx]
            y_values = y_values[valid_idx]
            z_values = z_values[valid_idx]
            
            logger.info(f"[PointCloud] Grid mode: {np.sum(valid_idx)}/{valid_mask.sum()} valid points after filtering")
            
            # 圧縮データ専用のデバッグ出力
            logger.debug(f"[PointCloud] Compressed grid stats - X: min={np.min(x_values):.2f}, max={np.max(x_values):.2f}")
            logger.debug(f"[PointCloud] Compressed grid stats - Y: min={np.min(y_values):.2f}, max={np.max(y_values):.2f}")
            logger.debug(f"[PointCloud] Compressed grid stats - Z: min={np.min(z_values):.2f}, max={np.max(z_values):.2f}")
            
            # 結果をスタック
            points = np.stack((x_values, y_values, z_values), axis=-1)
            logger.info(f"[PointCloud] Generated {points.shape[0]} points from compressed grid")
            return points
            
        else:
            # フル解像度深度マップの処理 (ベクトル化)
            h, w = depth_data.shape[:2]
            v_coords, u_coords = np.indices((h, w))
            
            valid_mask = depth_data > 0.01  # 有効な深度点のみを対象
            z_values = depth_data[valid_mask]
            
            if z_values.size == 0:
                logger.warning("[PointCloud] No valid depth values found")
                return np.empty((0, 3), dtype=np.float32)
            
            u_values = u_coords[valid_mask]
            v_values = v_coords[valid_mask]
            
            # 3D座標を計算
            x_cam = (u_values - cx) * z_values / fx
            y_cam = (v_values - cy) * z_values / fy
            
            # 異常値フィルタリング
            valid_idx = ~(np.isnan(x_cam) | np.isnan(y_cam) | np.isnan(z_values) | 
                          np.isinf(x_cam) | np.isinf(y_cam) | np.isinf(z_values))
            valid_idx &= (np.abs(x_cam) < 10) & (np.abs(y_cam) < 10) & (z_values < 20) & (z_values > 0)
            
            x_cam = x_cam[valid_idx]
            y_cam = y_cam[valid_idx]
            z_values = z_values[valid_idx]
            
            logger.info(f"[PointCloud] Full res mode: {valid_idx.sum()}/{valid_mask.sum()} points after filtering")
            
            # 結果のスタック
            points = np.stack((x_cam, y_cam, z_values), axis=-1)
            return points
        
    except Exception as e:
        logger.error(f"[PointCloud] Error in depth_to_point_cloud: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return np.empty((0, 3), dtype=np.float32)

def create_top_down_occupancy_grid(points, grid_resolution=GRID_RESOLUTION, 
                                  grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, 
                                  height_threshold=HEIGHT_THRESHOLD):
    """
    3D点群から天頂視点の占有グリッドを生成します。
    圧縮データにも対応した最適化バージョンです。

    Args:
        points (numpy.ndarray): 形状 (N, 3) の3D点群データ
        grid_resolution (float): グリッドの解像度（メートル/セル）
        grid_width (int): グリッドの幅（セル数）
        grid_height (int): グリッドの高さ（セル数）
        height_threshold (float): 通行可能と判定する高さの閾値（メートル）
    
    Returns:
        numpy.ndarray: 形状 (grid_height, grid_width) の占有グリッド
            0: 不明（データなし）
            1: 占有（障害物）
            2: 通行可能
    """
    try:
        logger.info(f"[OccGrid] Creating occupancy grid: resolution={grid_resolution}m, size={grid_width}x{grid_height} cells")
        
        # 初期化: すべてのセルを「不明」に設定
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # 空の点群チェック
        if points is None or not isinstance(points, np.ndarray) or points.size == 0:
            logger.warning("[OccGrid] Empty point cloud, returning default grid")
            return grid
        
        logger.debug(f"[OccGrid] Processing {points.shape[0]} points")
        logger.debug(f"[OccGrid] Point cloud data type: {points.dtype}")
        
        # グリッドの中心位置を計算（カメラ位置を基準に）
        grid_center_x = grid_width // 2
        grid_center_y = grid_height - 10  # カメラの少し前を中心とする
        
        # 点群データをグリッド座標に変換（ベクトル化処理）
        # X軸（左右）をグリッドの横方向にマッピング
        grid_x = np.round(points[:, 0] / grid_resolution + grid_center_x).astype(int)
        # Z軸（前後）をグリッドの縦方向にマッピング
        grid_y = grid_center_y - np.round(points[:, 2] / grid_resolution).astype(int)
        # Y軸（上下）は高さとして使用
        height = points[:, 1]
        
        # 処理前にグリッドの範囲を確認
        logger.debug(f"[OccGrid] Grid X range: {np.min(grid_x)} to {np.max(grid_x)}, Grid Y range: {np.min(grid_y)} to {np.max(grid_y)}")
        logger.debug(f"[OccGrid] Height range: {np.min(height)} to {np.max(height)}")
        
        # 高さの分布を確認（床と天井の検出に重要）
        height_percentiles = np.percentile(height, [5, 25, 50, 75, 95])
        logger.debug(f"[OccGrid] Height percentiles [5,25,50,75,95]: {height_percentiles}")
        
        # 高さの統計情報からしきい値を適応的に決定
        # 5パーセンタイルを床判定の基準に、75パーセンタイルを障害物判定の基準に
        adaptive_floor_threshold = height_percentiles[0] * 0.7  # 5パーセンタイルの70%を床の閾値に
        adaptive_obstacle_threshold = height_percentiles[3] * 0.5  # 75パーセンタイルの50%を障害物閾値に
        
        logger.info(f"[OccGrid] Using adaptive thresholds - floor: {adaptive_floor_threshold:.3f}m, obstacle: {adaptive_obstacle_threshold:.3f}m")
        
        # グリッド内の点のみを処理
        valid_idx = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
        valid_count = np.sum(valid_idx)
        logger.debug(f"[OccGrid] Valid points in grid: {valid_count}/{points.shape[0]} ({valid_count/points.shape[0]*100:.1f}%)")
        if np.sum(valid_idx) == 0:
            logger.warning("[OccGrid] No points fall within grid bounds")
            return grid
        
        grid_x = grid_x[valid_idx]
        grid_y = grid_y[valid_idx]
        height = height[valid_idx]
        
        logger.info(f"[OccGrid] {np.sum(valid_idx)} points within grid bounds")
        
        # オプション：より高速な処理のためにNumPyのベクトル化処理を使用
        # 各グリッドセルごとに最適な分類を決定
        
        # 圧縮データ用に最適化：点単位ではなくグリッドセル単位で処理
        # 各セルの座標とその高さ値をグループ化
        unique_cells = {}  # (x, y) -> [heights]
        
        # グリッドセルと高さ値のマッピングを作成
        for i, (x, y, h) in enumerate(zip(grid_x, grid_y, height)):
            cell_key = (x, y)
            if cell_key not in unique_cells:
                unique_cells[cell_key] = []
            unique_cells[cell_key].append(h)
        
        # 各セルの分類を決定
        logger.debug(f"[OccGrid] Processing {len(unique_cells)} unique grid cells")
        
        for (x, y), heights in unique_cells.items():
            # 複数の高さ値がある場合は統計量を計算
            heights_array = np.array(heights)
            min_height = np.min(heights_array)
            max_height = np.max(heights_array)
            
            # 高さの閾値を使って分類（圧縮データに合わせて最適化）
            # 適応的に決定された閾値を使用
            
            # 高さの中央値と標準偏差を計算（ノイズに強い分析）
            median_height = np.median(heights_array)
            height_std = np.std(heights_array)
            
            # 床判定（低い位置にある点）- 適応的閾値を使用
            if min_height < adaptive_floor_threshold:
                # 床（床面の点を含むセル）= 通行可能
                grid[y, x] = 2
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as FREE (floor): heights [{min_height:.2f} to {max_height:.2f}]")
            # 高さのばらつきが大きい場合は障害物（表面が不均一な物体）
            elif height_std > abs(adaptive_floor_threshold) * 0.5:
                grid[y, x] = 1
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as OBSTACLE (high variance: {height_std:.3f}): heights [{min_height:.2f} to {max_height:.2f}]")
            # 中央値が閾値範囲内にある場合は床
            elif median_height < adaptive_obstacle_threshold and median_height > adaptive_floor_threshold * 1.5:
                grid[y, x] = 2
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as FREE (median height): heights [{min_height:.2f} to {max_height:.2f}]")
            # 上部に点が集中している場合は通行可能（高い位置の物体）
            elif max_height > adaptive_obstacle_threshold * 3:
                grid[y, x] = 2
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as FREE (high object): heights [{min_height:.2f} to {max_height:.2f}]")
            else:
                # それ以外は障害物と判定
                grid[y, x] = 1
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as OBSTACLE (default): heights [{min_height:.2f} to {max_height:.2f}]")
        
        # グリッドの簡単な統計
        unknown_cells = np.sum(grid == 0)
        obstacle_cells = np.sum(grid == 1)
        free_cells = np.sum(grid == 2)
        logger.info(f"[OccGrid] Grid stats: unknown={unknown_cells}, obstacle={obstacle_cells}, free={free_cells}")
        
        return grid
        
    except Exception as e:
        logger.error(f"[OccGrid] Error in create_top_down_occupancy_grid: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return np.zeros((grid_height, grid_width), dtype=np.uint8)

def visualize_occupancy_grid(occupancy_grid, scale_factor=5):
    """
    占有グリッドを視覚化する関数。圧縮データに合わせて最適化。
    
    Args:
        occupancy_grid: 占有グリッド（0=不明、1=障害物、2=通行可能）
        scale_factor: 表示を拡大する係数
    
    Returns:
        可視化された画像
    """
    try:
        logger.info(f"[OccVis] Visualizing occupancy grid with shape {occupancy_grid.shape}, scale={scale_factor}")
        
        # グリッドチェック
        if occupancy_grid is None or not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.size == 0:
            logger.warning("[OccVis] Invalid occupancy grid")
            return np.zeros((240, 320, 3), dtype=np.uint8)
        
        # グリッドの統計を出力
        unique_values, counts = np.unique(occupancy_grid, return_counts=True)
        stats = {val: count for val, count in zip(unique_values, counts)}
        logger.debug(f"[OccVis] Occupancy grid stats: {stats}")
        
        # グリッドのサイズ
        grid_h, grid_w = occupancy_grid.shape
        
        # 小さいグリッドの場合は大きめのスケールファクターを使用
        if grid_h < 50 or grid_w < 50:
            logger.debug(f"[OccVis] Small grid detected, using larger scale factor: {scale_factor}")
        
        scaled_h = grid_h * scale_factor
        scaled_w = grid_w * scale_factor
        
        # 表示用のキャンバスを作成（RGB）
        visualization = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
        
        # 色の定義 (BGR) - 視認性を高めるためにコントラストを強調
        colors = {
            0: [50, 50, 50],     # 不明領域: 暗いグレー
            1: [0, 0, 255],      # 障害物: より明るい赤色
            2: [0, 255, 0]       # 通行可能: より明るい緑色
        }
        
        # グリッドの内容を描画（ベクトル化処理で高速化）
        # NumPy操作で画像全体を一度に作成
        for cell_value, color in colors.items():
            mask = occupancy_grid == cell_value
            if np.any(mask):
                # マスクを拡大
                expanded_mask = np.repeat(np.repeat(mask, scale_factor, axis=0), scale_factor, axis=1)
                
                # 色を適用
                for c_idx, c_val in enumerate(color):
                    visualization[:, :, c_idx][expanded_mask] = c_val
        
        # 中央に車両位置を示す点を描画
        # 元のグリッドでの中心位置 (圧縮グリッドに合わせて計算)
        orig_center_x, orig_center_y = grid_w // 2, grid_h - grid_h // 10
        
        # スケール後の中心位置 (セルの中心になるように調整)
        center_x = orig_center_x * scale_factor + scale_factor // 2
        center_y = orig_center_y * scale_factor + scale_factor // 2
        
        # 車両位置のマーカー
        marker_radius = max(3, scale_factor)
        cv2.circle(visualization, (center_x, center_y), marker_radius, [255, 255, 255], -1)
        
        # 進行方向の矢印
        arrow_length = max(10, scale_factor * 2)
        arrow_thickness = max(1, scale_factor // 2)
        cv2.arrowedLine(visualization,
                      (center_x, center_y),
                      (center_x, center_y - arrow_length),
                      [255, 255, 255],
                      arrow_thickness,
                      tipLength=0.3)
        
        # グリッド線を描画（視覚的な参考用）
        line_color = [50, 50, 50]  # 暗めのグレー
        line_thickness = 1
        
        # 小さいグリッド用に間隔を調整
        grid_spacing = max(1, min(5, grid_h // 5))
        
        # 水平線と垂直線を描画
        for i in range(0, grid_h + 1, grid_spacing):
            y = i * scale_factor
            cv2.line(visualization, (0, y), (scaled_w, y), line_color, line_thickness)
            
        for j in range(0, grid_w + 1, grid_spacing):
            x = j * scale_factor
            cv2.line(visualization, (x, 0), (x, scaled_h), line_color, line_thickness)
        
        # 1メートルスケールを表示（グリッド解像度を可視化）
        meter_text = "1m"
        # グリッドの解像度から1メートルあたりのピクセル数を計算
        grid_resolution = 0.1 * 20  # 圧縮グリッド用の解像度
        pixels_per_meter = scale_factor / grid_resolution
        meter_line_length = int(pixels_per_meter)
        
        # スケールバーを描画
        scale_bar_y = scaled_h - 30
        scale_bar_x = 20
        cv2.line(visualization, 
                (scale_bar_x, scale_bar_y), 
                (scale_bar_x + meter_line_length, scale_bar_y), 
                [200, 200, 200], 2)
        # スケールテキストを描画
        cv2.putText(visualization, meter_text, 
                   (scale_bar_x + meter_line_length // 2 - 10, scale_bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [200, 200, 200], 1)
        
        # セル統計情報を追加
        unknown_cells = np.sum(occupancy_grid == 0)
        obstacle_cells = np.sum(occupancy_grid == 1)
        free_cells = np.sum(occupancy_grid == 2)
        total_cells = grid_h * grid_w
        
        stats_text = f"Free: {free_cells}/{total_cells} ({free_cells/total_cells*100:.0f}%)"
        cv2.putText(visualization, stats_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, [30, 220, 30], 2)
        
        stats_text = f"Obstacle: {obstacle_cells}/{total_cells} ({obstacle_cells/total_cells*100:.0f}%)"
        cv2.putText(visualization, stats_text, (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 50, 220], 2)
        
        logger.info(f"[OccVis] Visualization complete: {visualization.shape}")
        return visualization
        
    except Exception as e:
        logger.error(f"[OccVis] Error in visualize_occupancy_grid: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return np.zeros((240, 320, 3), dtype=np.uint8)
        
# 末尾のテストコードを修正
# 問題のコード: h, w = absolute_depth.shape[:2] が存在

# 以下のように修正：
if __name__ == "__main__":
    # このブロックはモジュールが直接実行されたときのみ実行される
    import numpy as np
    
    # テスト用のダミー深度マップ
    test_depth = np.zeros((240, 320), dtype=np.float32)
    
    # 中央に円形の障害物を配置
    for i in range(240):
        for j in range(320):
            dist = np.sqrt((i-120)**2 + (j-160)**2)
            if dist < 50:
                test_depth[i, j] = 0.5  # 近い障害物
            else:
                test_depth[i, j] = 1.0  # 遠い背景
    
    # 点群に変換
    test_points = depth_to_point_cloud(test_depth, 500, 500)
    
    # 占有グリッドに変換
    test_grid = create_top_down_occupancy_grid(test_points, 0.05, 200, 200, 0.5)
    
    # 可視化
    test_vis = visualize_occupancy_grid(test_grid)
    
    # 画像を保存（必要に応じて）
    # import cv2
    # cv2.imwrite("test_topview.jpg", test_vis)
    
    print("Test completed successfully")