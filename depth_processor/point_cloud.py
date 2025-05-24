"""
Functions to generate point clouds from depth maps and create top-down view displays.
"""

import numpy as np
import cv2
import logging

# Import English text utilities
from english_text_utils import setup_matplotlib_english, cv2_put_english_text

# Logger setup
logger = logging.getLogger("kuma_depth_opt.point_cloud")
# --- ここから追加 ---
# logger のレベルを DEBUG に設定
logger.setLevel(logging.DEBUG)
# ハンドラが設定されていなければ、標準出力へのハンドラを追加
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# --- ここまで追加 ---
# logger.info("Logger for \'kuma_depth_opt.point_cloud\' INITIALIZED.") # Already initialized by getLogger

def depth_to_point_cloud(depth_data, camera_intrinsics: dict, is_grid_data: bool = False, grid_config: dict = None, original_image_dims: tuple = None):
    """
    Generate a 3D point cloud from depth data.
    Supports both high-resolution depth maps and compressed grid data.
    
    Args:
        depth_data (numpy.ndarray): Depth data. Full resolution map or compressed grid.
        camera_intrinsics (dict): Camera intrinsic parameters (fx, fy, cx, cy).
        is_grid_data (bool): If True, process depth_data as compressed grid.
        grid_config (dict, optional): Configuration for grid data, including 
                                      target_rows, target_cols. Required if is_grid_data is True.
        original_image_dims (tuple, optional): Dimensions (height, width) of the original image. 
                                             Required if is_grid_data is True and cx, cy 
                                             in camera_intrinsics are for the original image.

    Returns:
        numpy.ndarray: Point cloud data with shape (N, 3). Each point is [x, y, z].
                       Returns an empty array if there are no valid points.
    """
    try:
        fx = camera_intrinsics.get("fx")
        fy = camera_intrinsics.get("fy")
        cx = camera_intrinsics.get("cx")
        cy = camera_intrinsics.get("cy")

        if not all([fx, fy, cx, cy]):
            logger.error("[PointCloud] Missing camera intrinsic parameters (fx, fy, cx, cy).")
            return np.empty((0, 3), dtype=np.float32)

        min_depth_val = np.min(depth_data) if depth_data.size > 0 else 'N/A'
        max_depth_val = np.max(depth_data) if depth_data.size > 0 else 'N/A'
        logger.debug(f"[PointCloud] Input depth_data shape: {depth_data.shape}, range: {min_depth_val} to {max_depth_val}")
        logger.debug(f"[PointCloud] Camera params: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        if depth_data is None or depth_data.size == 0:
            logger.warning("[PointCloud] Error: Empty depth data provided.")
            return np.empty((0, 3), dtype=np.float32)
        
        points = np.empty((0, 3), dtype=np.float32)

        if is_grid_data:
            if grid_config is None:
                logger.error("[PointCloud] grid_config is required when is_grid_data is True.")
                return np.empty((0, 3), dtype=np.float32)
            
            if original_image_dims is None:
                logger.error("[PointCloud] original_image_dims is required for grid data when camera_intrinsics (cx, cy) are for the original image.")
                return np.empty((0, 3), dtype=np.float32)

            original_img_height, original_img_width = original_image_dims
            logger.debug(f"[PointCloud] Using original_image_dims: {original_img_width}x{original_img_height}")

            grid_rows_config = grid_config.get("target_rows")
            grid_cols_config = grid_config.get("target_cols")

            if not grid_rows_config or not grid_cols_config:
                logger.error("[PointCloud] target_rows or target_cols missing in grid_config.")
                return np.empty((0, 3), dtype=np.float32)
            
            current_grid_rows, current_grid_cols = depth_data.shape[:2]

            if current_grid_rows != grid_rows_config or current_grid_cols != grid_cols_config:
                logger.warning(f"[PointCloud] Depth data shape ({current_grid_rows}x{current_grid_cols}) does not match grid_config ({grid_rows_config}x{grid_cols_config}). Using actual depth_data.shape for grid dimensions.")

            logger.debug(f"[PointCloud] Grid mode with actual grid dimensions: rows={current_grid_rows}, cols={current_grid_cols}")
            
            r_indices, c_indices = np.indices((current_grid_rows, current_grid_cols))
            
            valid_mask = depth_data > 0.01 # Filter out very small or zero depth values
            valid_depth = depth_data[valid_mask]
            
            if valid_depth.size == 0:
                logger.warning("[PointCloud] No valid depth values (>0.01) in grid data.")
                return np.empty((0, 3), dtype=np.float32)
                
            valid_r_grid = r_indices[valid_mask]
            valid_c_grid = c_indices[valid_mask]
            
            # Calculate centers of the grid cells
            u_grid_centers = valid_c_grid + 0.5 
            v_grid_centers = valid_r_grid + 0.5
            
            # Map grid cell centers to original image pixel coordinates
            u_original = (u_grid_centers / current_grid_cols) * original_img_width
            v_original = (v_grid_centers / current_grid_rows) * original_img_height
            
            Z = valid_depth
            X = (u_original - cx) * Z / fx
            Y = (v_original - cy) * Z / fy
            
            points = np.stack((X, Y, Z), axis=-1)

        else: # Full resolution depth map
            if depth_data.ndim != 2:
                logger.error(f"[PointCloud] Full resolution depth data must be 2D, but got shape {depth_data.shape}")
                return np.empty((0, 3), dtype=np.float32)

            height, width = depth_data.shape
            
            c_indices, r_indices = np.meshgrid(np.arange(width), np.arange(height))

            valid_mask = depth_data > 0.01 
            valid_depth = depth_data[valid_mask]

            if valid_depth.size == 0:
                logger.warning("[PointCloud] No valid depth values (>0.01) in full-resolution data.")
                return np.empty((0, 3), dtype=np.float32)
            
            valid_c = c_indices[valid_mask] # u coordinates
            valid_r = r_indices[valid_mask] # v coordinates

            Z = valid_depth
            X = (valid_c - cx) * Z / fx
            Y = (valid_r - cy) * Z / fy
            
            points = np.stack((X, Y, Z), axis=-1)

        logger.debug(f"[PointCloud] Generated {points.shape[0]} points.")
        return points

    except Exception as e:
        logger.error(f"[PointCloud] Error generating point cloud: {e}", exc_info=True)
        return np.empty((0, 3), dtype=np.float32)

def create_top_down_occupancy_grid(points, grid_params):
    """
    Generate a top-down occupancy grid from a 3D point cloud.
    This is an optimized version that also supports compressed data.

    Args:
        points (numpy.ndarray): 3D point cloud data with shape (N, 3)
        grid_params : dict
            x_range : tuple (min_x, max_x)
                X-axis range
            z_range : tuple (min_z, max_z)
                Z-axis range
            resolution : float
                Grid cell size (meters)
            y_min : float
                Minimum height (Y coordinate) to be considered
            y_max : float
                Maximum height (Y Coordinate) to be considered
    
    Returns:
        numpy.ndarray: Occupancy grid with shape (grid_height, grid_width)
            0: Unknown (no data)
            1: Occupied (obstacle)
            2: Free to pass
    """
    try:
        x_min, x_max = grid_params["x_range"]
        z_min, z_max = grid_params["z_range"]
        grid_resolution = grid_params["resolution"]
        y_min_filter = grid_params.get("y_min_filter", -float('inf')) # Renamed from y_min to avoid conflict
        y_max_filter = grid_params.get("y_max_filter", float('inf')) # Renamed from y_max
        obstacle_height_threshold = grid_params.get("obstacle_height_threshold", 0.2) # Min height to be obstacle
        free_space_max_height = grid_params.get("free_space_max_height", 0.1) # Max height to be free space

        grid_height_cells = int(np.ceil((z_max - z_min) / grid_resolution))
        grid_width_cells = int(np.ceil((x_max - x_min) / grid_resolution))
        
        # Initialize grid: 0 for unknown, 1 for occupied, 2 for free
        occupancy_grid = np.zeros((grid_height_cells, grid_width_cells), dtype=np.uint8)
        # Store min height per cell to differentiate free vs obstacle
        min_height_grid = np.full((grid_height_cells, grid_width_cells), np.inf, dtype=np.float32)
        # Store max height per cell (less critical here but could be useful)
        # max_height_grid = np.full((grid_height_cells, grid_width_cells), -np.inf, dtype=np.float32)

        logger.info(f"[OccGrid] Creating {grid_width_cells}x{grid_height_cells} grid, res={grid_resolution}m")
        
        if points is None or points.size == 0:
            logger.warning("[OccGrid] Empty point cloud, returning empty grid.")
            return occupancy_grid

        # Filter points by height (Y coordinate in point cloud)
        # Y is typically vertical/height, X and Z are ground plane
        valid_height_mask = (points[:, 1] >= y_min_filter) & (points[:, 1] <= y_max_filter)
        filtered_points = points[valid_height_mask]

        if filtered_points.size == 0:
            logger.warning("[OccGrid] No points within Y filter range.")
            return occupancy_grid

        # Convert point coordinates to grid cell indices
        # X points -> grid_x (columns), Z points -> grid_z (rows)
        grid_x_indices = ((filtered_points[:, 0] - x_min) / grid_resolution).astype(int)
        grid_z_indices = ((filtered_points[:, 2] - z_min) / grid_resolution).astype(int)
        point_heights = filtered_points[:, 1]

        # Filter points that fall outside the defined grid
        valid_grid_mask = (grid_x_indices >= 0) & (grid_x_indices < grid_width_cells) & \
                          (grid_z_indices >= 0) & (grid_z_indices < grid_height_cells)
        
        grid_x_indices = grid_x_indices[valid_grid_mask]
        grid_z_indices = grid_z_indices[valid_grid_mask]
        point_heights = point_heights[valid_grid_mask]

        if grid_x_indices.size == 0:
            logger.warning("[OccGrid] No points fall within the grid boundaries after filtering.")
            return occupancy_grid

        # Populate min_height_grid
        # This uses a common trick for vectorized minimum: scatter points to a large array then reduce
        # A simpler loop might be clearer for fewer points, but this can be faster for many.
        # For loop approach (often clearer and sufficient unless performance is critical):
        for i in range(len(grid_x_indices)):
            gx, gz, ph = grid_x_indices[i], grid_z_indices[i], point_heights[i]
            min_height_grid[gz, gx] = min(min_height_grid[gz, gx], ph)
            # max_height_grid[gz, gx] = max(max_height_grid[gz, gx], ph)
            occupancy_grid[gz, gx] = 255 # Mark as having some data initially

        # Classify cells based on min_height_grid
        # Cells with points but min_height is low -> free space (value 2)
        # Cells with points and min_height is high -> obstacle (value 1)
        # Cells with no points (still 0) -> unknown
        
        # Mask for cells that received any points
        has_data_mask = occupancy_grid == 255 # Or min_height_grid != np.inf

        # Free cells: has data AND min_height is below free_space_max_height
        free_mask = has_data_mask & (min_height_grid <= free_space_max_height)
        occupancy_grid[free_mask] = 2

        # Obstacle cells: has data AND min_height is above obstacle_height_threshold
        # (and not already marked free, though logic should handle if thresholds overlap)
        # A point is an obstacle if its lowest part is above the obstacle_threshold.
        obstacle_mask = has_data_mask & (min_height_grid > obstacle_height_threshold)
        occupancy_grid[obstacle_mask] = 1
        
        # Any remaining cells with data that didn't meet free/obstacle criteria (e.g. between thresholds)
        # could be marked as unknown or a specific intermediate state. For now, they might remain 255.
        # Let's ensure they are at least marked as occupied if not free.
        intermediate_mask = has_data_mask & (occupancy_grid == 255)
        occupancy_grid[intermediate_mask] = 1 # Default to obstacle if ambiguous and has data

        logger.info(f"[OccGrid] Grid populated. Free: {np.sum(occupancy_grid == 2)}, Obstacle: {np.sum(occupancy_grid == 1)}, Unknown: {np.sum(occupancy_grid == 0)}")
        return occupancy_grid
        
    except Exception as e:
        logger.error(f"[OccGrid] Error in create_top_down_occupancy_grid: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        # Return an empty grid of expected type upon error, using calculated dimensions if possible
        try:
            ghc = int(np.ceil((grid_params["z_range"][1] - grid_params["z_range"][0]) / grid_params["resolution"]))
            gwc = int(np.ceil((grid_params["x_range"][1] - grid_params["x_range"][0]) / grid_params["resolution"]))
            return np.zeros((ghc, gwc), dtype=np.uint8)
        except:
            return np.zeros((10,10), dtype=np.uint8) # Fallback fixed size


def visualize_occupancy_grid(occupancy_grid, scale_factor=10):
    """
    Visualize the occupancy grid.
    
    Args:
        occupancy_grid (numpy.ndarray): Occupancy grid (0=unknown, 1=obstacle, 2=free)
        scale_factor (int): Factor to enlarge the display
    
    Returns:
        numpy.ndarray: Visualized image (BGR)
    """
    try:
        if occupancy_grid is None or occupancy_grid.size == 0:
            logger.warning("[OccVis] Empty or invalid occupancy grid for visualization.")
            return np.full((100, 100, 3), [128,128,128], dtype=np.uint8) # Gray image

        grid_h, grid_w = occupancy_grid.shape
        logger.info(f"[OccVis] Visualizing {grid_w}x{grid_h} grid with scale {scale_factor}")

        scaled_h, scaled_w = grid_h * scale_factor, grid_w * scale_factor
        visualization = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)

        colors = {
            0: [80, 80, 80],    # Unknown: Dark Gray
            1: [0, 0, 200],    # Obstacle: Red
            2: [0, 200, 0],    # Free: Green
            255: [200,0,200] # Magenta for any unexpected values (e.g. if 255 was used as intermediate)
        }

        for val, color in colors.items():
            mask = occupancy_grid == val
            # Efficiently color using repeat and boolean indexing
            visualization[np.repeat(mask, scale_factor, axis=0).repeat(scale_factor, axis=1)] = color
        
        # Optional: Add grid lines for clarity
        for i in range(0, scaled_w, scale_factor):
            cv2.line(visualization, (i, 0), (i, scaled_h), (50,50,50), 1)
        for i in range(0, scaled_h, scale_factor):
            cv2.line(visualization, (0, i), (scaled_w, i), (50,50,50), 1)

        # Vehicle position marker (example: center bottom)
        center_x = scaled_w // 2
        center_y = scaled_h - (scale_factor // 2) # A bit up from the very bottom edge
        if center_y < 0 : center_y = scaled_h //2 # handle very small grids
        cv2.circle(visualization, (center_x, center_y), scale_factor // 2 +1 , (255, 255, 255), -1) # White circle
        # Arrow for direction (pointing "up" in the image, which is typically forward)
        cv2.arrowedLine(visualization, (center_x, center_y),
                        (center_x, center_y - scale_factor * 2 if center_y - scale_factor * 2 > 0 else 0),
                        (255,255,255), max(1,scale_factor//4), tipLength=0.4)

        logger.debug("[OccVis] Visualization generated.")
        return visualization

    except Exception as e:
        logger.error(f"[OccVis] Error in visualize_occupancy_grid: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return np.full((100, 100, 3), [0,0,0] , dtype=np.uint8) # Black image on error

# Modified test code
# Fixed previous issue: h, w = absolute_depth.shape[:2]

# Modified as follows:
if __name__ == "__main__":
    # This block is executed only when the module is run directly
    import numpy as np
    
    # Dummy depth map for testing
    test_depth = np.zeros((240, 320), dtype=np.float32)
    
    # Place a circular obstacle in the center
    for i in range(240):
        for j in range(320):
            dist = np.sqrt((i-120)**2 + (j-160)**2)
            if dist < 50:
                test_depth[i, j] = 0.5  # Closer obstacle
            else:
                test_depth[i, j] = 1.0  # Distant background
    
    # Convert to point cloud
    test_points = depth_to_point_cloud(test_depth, {"fx": 500, "fy": 500, "cx": 160, "cy": 120})
    
    # Convert to occupancy grid
    test_grid = create_top_down_occupancy_grid(test_points, 0.05, 200, 200, 0.5)
    
    # Visualize
    test_vis = visualize_occupancy_grid(test_grid)
    
    # Save image (if needed)
    # import cv2
    # cv2.imwrite("test_topview.jpg", test_vis)
    
    print("Test completed successfully")

# --- ここから追加 ---
# Refactored test code to match new function signatures and add more comprehensive tests

if __name__ == "__main__":
    # Configure logger for testing this script directly
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Testing point_cloud.py functions...")

    # --- Test Case 1: Full Resolution Depth --- 
    mock_depth_full_res = np.ones((480, 640), dtype=np.float32) * 2.5 # 2.5 meters depth
    # Add some variation
    mock_depth_full_res[100:200, 200:300] = 1.0 # Closer object
    mock_camera_intrinsics_full = {
        "fx": 500.0, "fy": 500.0, 
        "cx": 319.5, "cy": 239.5 # Center of 640x480 image
    }
    
    logger.info("\n--- Testing Full Resolution --- ")
    pc_full = depth_to_point_cloud(mock_depth_full_res, mock_camera_intrinsics_full, is_grid_data=False)
    if pc_full.size > 0:
        logger.info(f"Full res point cloud generated: {pc_full.shape[0]} points.")
        logger.debug(f"Sample points (full res):\n{pc_full[:3]}")
        logger.debug(f"X range: {np.min(pc_full[:,0]):.2f} to {np.max(pc_full[:,0]):.2f}")
        logger.debug(f"Y range: {np.min(pc_full[:,1]):.2f} to {np.max(pc_full[:,1]):.2f}")
        logger.debug(f"Z range: {np.min(pc_full[:,2]):.2f} to {np.max(pc_full[:,2]):.2f}")
    else:
        logger.error("Full res point cloud generation FAILED.")

    # --- Test Case 2: Grid Compressed Depth --- 
    mock_depth_grid = np.ones((12, 16), dtype=np.float32) * 3.0 # 3.0 meters depth for grid cells
    mock_depth_grid[3:6, 4:8] = 0.8 # Closer group of cells
    
    # IMPORTANT: For grid data, cx and cy should ideally be relative to the grid dimensions.
    # If original cx,cy were 319.5 for a 640 wide image, and grid is 16 cols,
    # new cx_grid could be (319.5 / 640) * 16 = 7.9875 approx 7.5 (center of 16 cols)
    # Similarly for cy_grid: (239.5 / 480) * 12 = 5.9875 approx 5.5 (center of 12 rows)
    # fx_grid = fx_orig * (grid_cols / orig_cols)
    # fy_grid = fy_orig * (grid_rows / orig_rows)
    # However, the problem description implies sending PointCloud, so the conversion uses the *original* camera intrinsics
    # but applies them to the *grid cell centers* mapped back to the original image plane if necessary.
    # The current implementation of depth_to_point_cloud for grid data assumes fx,fy,cx,cy are for the grid itself.
    # Let's define intrinsics for the grid for this test, assuming cx,cy are grid centers.
    mock_camera_intrinsics_grid = {
        "fx": 16.0, "fy": 12.0, # Example: if each grid cell was 1 unit FoV
        "cx": 7.5,  "cy": 5.5   # Center of 16x12 grid (0-indexed)
    }
    mock_grid_config = {"target_rows": 12, "target_cols": 16}

    logger.info("\n--- Testing Grid Compressed Data --- ")
    pc_grid = depth_to_point_cloud(mock_depth_grid, mock_camera_intrinsics_grid, 
                                   is_grid_data=True, grid_config=mock_grid_config)
    if pc_grid.size > 0:
        logger.info(f"Grid point cloud generated: {pc_grid.shape[0]} points.")
        logger.debug(f"Sample points (grid):\n{pc_grid[:3]}")
        logger.debug(f"X range: {np.min(pc_grid[:,0]):.2f} to {np.max(pc_grid[:,0]):.2f}")
        logger.debug(f"Y range: {np.min(pc_grid[:,1]):.2f} to {np.max(pc_grid[:,1]):.2f}")
        logger.debug(f"Z range: {np.min(pc_grid[:,2]):.2f} to {np.max(pc_grid[:,2]):.2f}")
    else:
        logger.error("Grid point cloud generation FAILED.")

    # --- Test Case 3: Empty depth data ---
    logger.info("\n--- Testing Empty Depth Data --- ")
    pc_empty = depth_to_point_cloud(np.array([]), mock_camera_intrinsics_full)
    if pc_empty.size == 0:
        logger.info("Empty depth data test PASSED (returned empty point cloud).")
    else:
        logger.error("Empty depth data test FAILED.")

    # --- Test Case 4: Grid data with no valid points ---
    logger.info("\n--- Testing Grid Data with No Valid Points --- ")
    mock_depth_grid_invalid = np.zeros((12,16), dtype=np.float32) # All zero depth
    pc_grid_invalid = depth_to_point_cloud(mock_depth_grid_invalid, mock_camera_intrinsics_grid, 
                                           is_grid_data=True, grid_config=mock_grid_config)
    if pc_grid_invalid.size == 0:
        logger.info("Grid data with no valid points test PASSED.")
    else:
        logger.error("Grid data with no valid points test FAILED.")

    # (Keep create_top_down_occupancy_grid and visualize_occupancy_grid if they are used by other parts or for testing)
    # For example, to test occupancy grid generation:
    # if pc_grid.size > 0:
    #     logger.info("\n--- Testing Occupancy Grid Generation (from grid PC) --- ")
    #     mock_grid_params_occ = {
    #         "x_range": (-5, 5), "z_range": (0, 10), "resolution": 0.1,
    #         "y_min": -1.0, "y_max": 1.0, "obstacle_threshold": 0.2
    #     }
    #     occupancy_map = create_top_down_occupancy_grid(pc_grid, mock_grid_params_occ)
    #     logger.info(f"Occupancy grid generated with shape: {occupancy_map.shape}")
    #     # visualize_occupancy_grid(occupancy_map) # If you want to see it (requires cv2 window)

    logger.info("\nPoint_cloud.py tests finished.")