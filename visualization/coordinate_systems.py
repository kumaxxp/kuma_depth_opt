import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# 英語表示用の設定
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

def create_comprehensive_algorithm_visualization():
    """
    アルゴリズム仕様書に対応した包括的な座標系と変換プロセスの可視化
    """
    # フィギュアのセットアップ（2x2のサブプロット）
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. カメラ座標系とピンホールモデル
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    _plot_camera_coordinate_system(ax1)
    
    # 2. 深度変換アルゴリズムの可視化
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_depth_conversion_algorithm(ax2)
    
    # 3. グリッド圧縮と座標変換
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_grid_compression_mapping(ax3)
    
    # 4. トップダウン変換
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_topdown_transformation(ax4)
    
    # メインタイトル
    fig.suptitle('Obstacle Detection System: Coordinate Systems & Algorithms', 
                 fontsize=16, fontweight='bold')
    
    # 画像として保存
    plt.savefig('comprehensive_visualization.png', dpi=300, bbox_inches='tight')
    print("Comprehensive visualization saved as 'comprehensive_visualization.png'")
    
    plt.show()
    return fig

def _plot_camera_coordinate_system(ax):
    """カメラ座標系とピンホールカメラモデルの可視化"""
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
    
    # ビューアングルを調整：Z軸を奥行方向、Y軸を縦方向に（人間が調整した見やすい角度）
    ax.view_init(elev=-30, azim=-90-70, roll=0+80)  # 人間が調整した最適な角度
    
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
    
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'k-', linewidth=2)
    
    # 光学中心と焦点距離の説明
    ax.scatter(0, 0, image_distance, color='orange', s=100, label='Optical Center (cx,cy)')
    
    # 投影線の例（標準的なカメラ座標系）
    world_points = np.array([
        [0.5, 0.3, 2.5],    # 右上の点
        [-0.4, -0.2, 2.8],  # 左下の点
        [0.3, -0.4, 3.0]    # 右下の点
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
    ax.text2D(0.02, 0.98, 'Pinhole Camera Model:\nX = (u - cx) * Z / fx\nY = (v - cy) * Z / fy\n\nCoordinate System:\nX: Right, Y: Down, Z: Forward', 
              transform=ax.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 軸ラベルを明確に
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Forward into scene)')
    ax.set_title('Camera Coordinate System\n& Pinhole Model')
    ax.legend(fontsize=8)
    
    # 軸の範囲を調整
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 3])
    
    # 軸の比率を等しく保つ
    ax.set_box_aspect([1,1,1.5])  # Z軸を少し長めに表示

def _plot_depth_conversion_algorithm(ax):
    """深度変換アルゴリズムの可視化"""
    # ダミーの相対深度データ
    x = np.linspace(0, 1, 100)
    relative_depth = 0.3 + 0.4 * np.sin(2 * np.pi * x * 3) * np.exp(-x * 2)
    
    # パーセンタイル計算
    p5 = np.percentile(relative_depth, 5)
    p95 = np.percentile(relative_depth, 95)
    
    # 正規化差分
    diff = p95 - relative_depth
    normalized_diff = np.clip(diff / (p95 - p5), 0, 1)
    
    # 絶対深度（scaling_factor=10の例）
    scaling_factor = 10.0
    absolute_depth = 0.5 + normalized_diff * scaling_factor
    
    # 相対深度のプロット
    ax.plot(x, relative_depth, 'b-', linewidth=2, label='Raw Relative Depth')
    ax.axhline(y=p5, color='red', linestyle='--', alpha=0.7, label=f'5th percentile: {p5:.3f}')
    ax.axhline(y=p95, color='red', linestyle='--', alpha=0.7, label=f'95th percentile: {p95:.3f}')
    ax.fill_between(x, p5, p95, alpha=0.2, color='yellow', label='Effective Range')
    
    ax.set_xlabel('Sample Index (normalized)')
    ax.set_ylabel('Relative Depth', color='blue')
    ax.set_title('Depth Conversion Algorithm')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 絶対深度用の第二Y軸
    ax2 = ax.twinx()
    ax2.plot(x, absolute_depth, 'g-', linewidth=2, label='Absolute Depth (m)')
    ax2.set_ylabel('Absolute Depth (m)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right', fontsize=8)
    
    # 変換式の表示
    formula_text = ('Conversion Steps:\n'
                   '1. Extract valid values (>0.01)\n'
                   '2. Calculate percentiles (5th, 95th)\n'
                   '3. Normalize: diff = (p95 - depth) / (p95 - p5)\n'
                   '4. Scale: abs_depth = 0.5 + diff * scaling_factor\n'
                   '5. Clip to [0.1m, 50m]')
    
    ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            verticalalignment='bottom')

def _plot_grid_compression_mapping(ax):
    """グリッド圧縮と座標マッピングの可視化"""
    # 元画像のグリッド（例：640x480）
    original_width, original_height = 640, 480
    grid_cols, grid_rows = 8, 6
    
    # グリッドセルのサイズ
    cell_width = original_width // grid_cols
    cell_height = original_height // grid_rows
    
    # 画像座標系の表示（上下反転して通常の画像表示に合わせる）
    ax.set_xlim(0, original_width)
    ax.set_ylim(original_height, 0)  # Y軸を反転
    
    # グリッド線の描画
    for i in range(grid_rows + 1):
        y = i * cell_height
        ax.axhline(y=y, color='blue', linewidth=1)
    
    for j in range(grid_cols + 1):
        x = j * cell_width
        ax.axvline(x=x, color='blue', linewidth=1)
    
    # いくつかのセルをハイライト
    highlight_cells = [(1, 2), (3, 4), (5, 1)]
    colors = ['red', 'green', 'orange']
    
    for idx, (row, col) in enumerate(highlight_cells):
        x_start = col * cell_width
        y_start = row * cell_height
        
        # セルの塗りつぶし
        rect = Rectangle((x_start, y_start), cell_width, cell_height,
                        facecolor=colors[idx], alpha=0.3, edgecolor=colors[idx], linewidth=2)
        ax.add_patch(rect)
        
        # セル中心の計算と表示
        center_u = x_start + cell_width * 0.5
        center_v = y_start + cell_height * 0.5
        ax.plot(center_u, center_v, 'ko', markersize=8)
        
        # グリッド座標の表示
        ax.text(center_u, center_v - 15, f'Grid({row},{col})', 
                ha='center', fontsize=8, fontweight='bold')
        
        # 元画像座標の表示
        ax.text(center_u, center_v + 15, f'Image({center_u:.1f},{center_v:.1f})', 
                ha='center', fontsize=8)
    
    # 座標変換式の表示
    transform_text = ('Grid to Image Mapping:\n'
                     'u_original = (u_grid + 0.5) / grid_cols * img_width\n'
                     'v_original = (v_grid + 0.5) / grid_rows * img_height\n\n'
                     'Camera Parameter Scaling:\n'
                     'fx_scaled = fx * (grid_cols / img_width)\n'
                     'fy_scaled = fy * (grid_rows / img_height)')
    
    ax.text(0.02, 0.98, transform_text, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            verticalalignment='top')
    
    ax.set_xlabel('Image X (u)')
    ax.set_ylabel('Image Y (v)')
    ax.set_title(f'Grid Compression Mapping\n{original_width}x{original_height} → {grid_cols}x{grid_rows}')
    ax.set_aspect('equal')

def _plot_topdown_transformation(ax):
    """トップダウン変換の可視化"""
    # 3Dポイントクラウドの例（カメラ座標系）
    np.random.seed(42)
    n_points = 100
    
    # カメラ座標系でのポイント生成
    camera_x = np.random.uniform(-2, 2, n_points)  # 左右
    camera_y = np.random.uniform(-1, 1, n_points)  # 上下（高さ）
    camera_z = np.random.uniform(1, 5, n_points)   # 前後（距離）
    
    # 床レベルの点（Y ≈ 0）
    floor_mask = np.abs(camera_y) < 0.2
    
    # 障害物レベルの点（Y > 0.3）
    obstacle_mask = camera_y > 0.3
    
    # トップダウンビュー（X-Z平面）
    # 色分け：青=床、赤=障害物、灰=その他
    colors = np.where(floor_mask, 'blue',
                     np.where(obstacle_mask, 'red', 'gray'))
    
    sizes = np.where(floor_mask, 20,
                    np.where(obstacle_mask, 40, 10))
    
    # ポイントクラウドの散布図
    scatter = ax.scatter(camera_x, camera_z, c=colors, s=sizes, alpha=0.7)
    
    # 占有グリッドの表示
    grid_resolution = 0.5  # 0.5m解像度
    x_range = (-3, 3)
    z_range = (0, 6)
    
    # グリッド線
    x_grid = np.arange(x_range[0], x_range[1] + grid_resolution, grid_resolution)
    z_grid = np.arange(z_range[0], z_range[1] + grid_resolution, grid_resolution)
    
    for x in x_grid:
        ax.axvline(x=x, color='lightgray', linewidth=0.5, alpha=0.5)
    for z in z_grid:
        ax.axhline(y=z, color='lightgray', linewidth=0.5, alpha=0.5)
    
    # サンプル占有グリッドセル
    sample_cells = [
        (1.0, 2.0, 'free', 'lightblue'),      # 自由空間
        (-0.5, 3.5, 'obstacle', 'lightcoral'), # 障害物
        (1.5, 4.0, 'unknown', 'lightgray')     # 不明
    ]
    
    for x_center, z_center, cell_type, color in sample_cells:
        rect = Rectangle((x_center - grid_resolution/2, z_center - grid_resolution/2), 
                        grid_resolution, grid_resolution,
                        facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x_center, z_center, cell_type, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # カメラ位置の表示
    ax.plot(0, 0, 'ks', markersize=12, label='Camera Position')
    ax.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 座標軸ラベル
    ax.set_xlabel('X: Left ← → Right (m)')
    ax.set_ylabel('Z: Camera → Forward (m)')
    ax.set_title('Top-Down View Transformation\n(Camera XY → Grid XY, Camera Y → Height)')
    
    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Floor (Y≈0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Obstacle (Y>0.3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='Other'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=8, label='Free Cell'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral', markersize=8, label='Obstacle Cell'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markersize=8, label='Unknown Cell')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # 変換説明
    transform_text = ('Coordinate Transformation:\n'
                     'Grid_X = Camera_X\n'
                     'Grid_Y = Camera_Z\n'
                     'Height = Camera_Y\n\n'
                     'Occupancy Classification:\n'
                     'Free: height ≤ 0.1m\n'
                     'Obstacle: height ≥ 0.2m\n'
                     'Unknown: 0.1m < height < 0.2m')
    
    ax.text(0.02, 0.02, transform_text, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9),
            verticalalignment='bottom')
    
    ax.set_xlim(x_range)
    ax.set_ylim(z_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def create_individual_visualizations():
    """
    個別の可視化を作成（デバッグや詳細確認用）
    """
    print("Creating individual visualizations...")
    
    # 1. カメラ座標系
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    _plot_camera_coordinate_system(ax1)
    plt.savefig('camera_coordinate_system.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("→ camera_coordinate_system.png saved")
    
    # 2. 深度変換アルゴリズム
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = fig2.add_subplot(111)
    _plot_depth_conversion_algorithm(ax2)
    plt.savefig('depth_conversion_algorithm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("→ depth_conversion_algorithm.png saved")
    
    # 3. グリッド圧縮
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111)
    _plot_grid_compression_mapping(ax3)
    plt.savefig('grid_compression_mapping.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("→ grid_compression_mapping.png saved")
    
    # 4. トップダウン変換
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111)
    _plot_topdown_transformation(ax4)
    plt.savefig('topdown_transformation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("→ topdown_transformation.png saved")
    
    print("All individual visualizations completed!")

def create_custom_visualization(scaling_factor=10.0, grid_cols=8, grid_rows=6, 
                               grid_resolution=0.5, height_threshold=0.2):
    """
    カスタムパラメータでの可視化
    
    Args:
        scaling_factor: 深度変換のスケーリング係数
        grid_cols: グリッド列数
        grid_rows: グリッド行数
        grid_resolution: 占有グリッドの解像度(m)
        height_threshold: 障害物判定の高さ閾値(m)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # カスタムパラメータを使用した可視化
    # （実際のアルゴリズムではこれらのパラメータが重要な役割を果たします）
    
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    _plot_camera_coordinate_system(ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_depth_conversion_algorithm_custom(ax2, scaling_factor)
    
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_grid_compression_mapping_custom(ax3, grid_cols, grid_rows)
    
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_topdown_transformation_custom(ax4, grid_resolution, height_threshold)
    
    fig.suptitle(f'Custom Visualization (scale={scaling_factor}, grid={grid_cols}x{grid_rows}, '
                f'res={grid_resolution}m, h_thresh={height_threshold}m)', 
                 fontsize=14, fontweight='bold')
    
    filename = f'custom_vis_s{scaling_factor}_g{grid_cols}x{grid_rows}_r{grid_resolution}_h{height_threshold}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Custom visualization saved as '{filename}'")
    plt.close()

def _plot_depth_conversion_algorithm_custom(ax, scaling_factor):
    """カスタムスケーリング係数での深度変換可視化"""
    x = np.linspace(0, 1, 100)
    relative_depth = 0.3 + 0.4 * np.sin(2 * np.pi * x * 3) * np.exp(-x * 2)
    
    p5 = np.percentile(relative_depth, 5)
    p95 = np.percentile(relative_depth, 95)
    diff = p95 - relative_depth
    normalized_diff = np.clip(diff / (p95 - p5), 0, 1)
    absolute_depth = 0.5 + normalized_diff * scaling_factor
    
    ax.plot(x, relative_depth, 'b-', linewidth=2, label='Raw Relative Depth')
    ax.axhline(y=p5, color='red', linestyle='--', alpha=0.7, label=f'5th percentile: {p5:.3f}')
    ax.axhline(y=p95, color='red', linestyle='--', alpha=0.7, label=f'95th percentile: {p95:.3f}')
    ax.fill_between(x, p5, p95, alpha=0.2, color='yellow', label='Effective Range')
    
    ax.set_xlabel('Sample Index (normalized)')
    ax.set_ylabel('Relative Depth', color='blue')
    ax.set_title(f'Depth Conversion (Scaling Factor: {scaling_factor})')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(x, absolute_depth, 'g-', linewidth=3, label=f'Absolute Depth (max: {np.max(absolute_depth):.1f}m)')
    ax2.set_ylabel('Absolute Depth (m)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right', fontsize=8)

def _plot_grid_compression_mapping_custom(ax, grid_cols, grid_rows):
    """カスタムグリッドサイズでの圧縮マッピング可視化"""
    original_width, original_height = 640, 480
    cell_width = original_width // grid_cols
    cell_height = original_height // grid_rows
    
    ax.set_xlim(0, original_width)
    ax.set_ylim(original_height, 0)
    
    for i in range(grid_rows + 1):
        y = i * cell_height
        ax.axhline(y=y, color='blue', linewidth=1)
    
    for j in range(grid_cols + 1):
        x = j * cell_width
        ax.axvline(x=x, color='blue', linewidth=1)
    
    # いくつかのセルをハイライト（グリッドサイズに応じて調整）
    max_highlights = min(3, grid_rows * grid_cols // 2)  # グリッドサイズに応じてハイライト数を調整
    
    if grid_rows >= 3 and grid_cols >= 3:
        # 標準的なハイライト位置
        highlight_cells = [(1, 1), (grid_rows//2, grid_cols//2)]
        if grid_rows > 3 and grid_cols > 3:
            highlight_cells.append((grid_rows-2, grid_cols-2))
    else:
        # 小さなグリッドの場合
        highlight_cells = [(0, 1), (1, 0)]
        if grid_rows > 2 and grid_cols > 2:
            highlight_cells.append((grid_rows-1, grid_cols-1))
    
    colors = ['red', 'green', 'orange']
    
    for idx, (row, col) in enumerate(highlight_cells[:len(colors)]):
        if row < grid_rows and col < grid_cols:  # 範囲チェック
            x_start = col * cell_width
            y_start = row * cell_height
            
            # セルの塗りつぶし
            rect = Rectangle((x_start, y_start), cell_width, cell_height,
                            facecolor=colors[idx], alpha=0.3, edgecolor=colors[idx], linewidth=2)
            ax.add_patch(rect)
            
            # セル中心の計算と表示
            center_u = x_start + cell_width * 0.5
            center_v = y_start + cell_height * 0.5
            ax.plot(center_u, center_v, 'ko', markersize=8)
            
            # テキスト位置をセルサイズに応じて調整
            text_offset_v = max(15, cell_height * 0.2)
            
            # グリッド座標の表示
            ax.text(center_u, center_v - text_offset_v, f'Grid({row},{col})', 
                    ha='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            # 元画像座標の表示
            ax.text(center_u, center_v + text_offset_v, f'Image({center_u:.1f},{center_v:.1f})', 
                    ha='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # 座標変換式の表示
    transform_text = ('Grid to Image Mapping:\n'
                     'u_original = (u_grid + 0.5) / grid_cols * img_width\n'
                     'v_original = (v_grid + 0.5) / grid_rows * img_height\n\n'
                     'Camera Parameter Scaling:\n'
                     'fx_scaled = fx * (grid_cols / img_width)\n'
                     'fy_scaled = fy * (grid_rows / img_height)')
    
    ax.text(0.7, 0.98, transform_text, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9),
            verticalalignment='top')
    
    # 圧縮率の計算と表示
    compression_ratio = (grid_cols * grid_rows) / (original_width * original_height)
    compression_text = (f'Compression Ratio: {compression_ratio:.6f}\n'
                       f'Original: {original_width}x{original_height} = {original_width*original_height:,} pixels\n'
                       f'Compressed: {grid_cols}x{grid_rows} = {grid_cols*grid_rows} cells\n'
                       f'Cell Size: {cell_width}x{cell_height} pixels')
    
    ax.text(0.02, 0.5, compression_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            verticalalignment='center')
    
    ax.set_xlabel('Image X (u)')
    ax.set_ylabel('Image Y (v)')
    ax.set_title(f'Custom Grid Compression: {grid_cols}x{grid_rows}')
    ax.set_aspect('equal')

def _plot_topdown_transformation_custom(ax, grid_resolution, height_threshold):
    """カスタムパラメータでのトップダウン変換可視化"""
    np.random.seed(42)
    n_points = 150
    
    camera_x = np.random.uniform(-3, 3, n_points)
    camera_y = np.random.uniform(-0.5, 1.5, n_points)
    camera_z = np.random.uniform(1, 6, n_points)
    
    # カスタム高さ閾値での分類
    floor_mask = camera_y < height_threshold * 0.5
    obstacle_mask = camera_y > height_threshold
    uncertain_mask = ~floor_mask & ~obstacle_mask
    
    colors = np.where(floor_mask, 'blue',
                     np.where(obstacle_mask, 'red', 'orange'))
    
    sizes = np.where(floor_mask, 15,
                    np.where(obstacle_mask, 50, 25))
    
    scatter = ax.scatter(camera_x, camera_z, c=colors, s=sizes, alpha=0.7)
    
    # カスタム解像度でのグリッド
    x_range = (-4, 4)
    z_range = (0, 7)
    
    x_grid = np.arange(x_range[0], x_range[1] + grid_resolution, grid_resolution)
    z_grid = np.arange(z_range[0], z_range[1] + grid_resolution, grid_resolution)
    
    for x in x_grid:
        ax.axvline(x=x, color='lightgray', linewidth=0.5, alpha=0.5)
    for z in z_grid:
        ax.axhline(y=z, color='lightgray', linewidth=0.5, alpha=0.5)
    
    # グリッド統計
    grid_count_x = len(x_grid) - 1
    grid_count_z = len(z_grid) - 1
    total_cells = grid_count_x * grid_count_z
    
    ax.plot(0, 0, 'ks', markersize=12, label='Camera Position')
    ax.arrow(0, 0, 0, 1, head_width=0.15, head_length=0.15, fc='black', ec='black')
    
    ax.set_xlabel('X: Left ← → Right (m)')
    ax.set_ylabel('Z: Camera → Forward (m)')
    ax.set_title(f'Custom Top-Down View (res: {grid_resolution}m, h_thresh: {height_threshold}m)')
    
    # 統計情報
    stats_text = (f'Grid Statistics:\n'
                 f'Resolution: {grid_resolution}m\n'
                 f'Grid Size: {grid_count_x}x{grid_count_z} = {total_cells} cells\n'
                 f'Height Threshold: {height_threshold}m\n'
                 f'Points: Floor={np.sum(floor_mask)}, '
                 f'Obstacle={np.sum(obstacle_mask)}, '
                 f'Uncertain={np.sum(uncertain_mask)}')
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9),
            verticalalignment='bottom')
    
    ax.set_xlim(x_range)
    ax.set_ylim(z_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    # メイン可視化
    print("Creating comprehensive visualization...")
    create_comprehensive_algorithm_visualization()
    
    # 個別可視化
    print("\nCreating individual visualizations...")
    create_individual_visualizations()
    
    # カスタム可視化の例
    print("\nCreating custom visualization examples...")
    create_custom_visualization(scaling_factor=5.0, grid_cols=4, grid_rows=3)  # 低解像度
    create_custom_visualization(scaling_factor=15.0, grid_cols=16, grid_rows=12, 
                              grid_resolution=0.25, height_threshold=0.15)  # 高解像度
    
    print("\nAll visualizations completed! Check the generated PNG files.")