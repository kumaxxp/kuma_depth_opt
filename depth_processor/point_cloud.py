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
logger.setLevel(logging.DEBUG)
# Add handler to standard output if not already set
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.info("Logger for 'kuma_depth_opt.point_cloud' INITIALIZED.")

# Default parameters
GRID_RESOLUTION = 0.06  # meters/cell
GRID_WIDTH = 100        # number of cells in the horizontal direction
GRID_HEIGHT = 100       # number of cells in the vertical direction
HEIGHT_THRESHOLD = 0.3  # height threshold for passability judgment (meters)
MAX_DEPTH = 6.0         # maximum depth (meters)

def depth_to_point_cloud(depth_data, fx, fy, cx, cy,
                         original_height=None, original_width=None,
                         is_grid_data=False, grid_rows=None, grid_cols=None):
    """
    Generate a 3D point cloud from depth data.
    Supports both high-resolution depth maps and compressed grid data.
    
    Args:
        depth_data (numpy.ndarray): Depth data. Full resolution map or compressed grid.
        fx (float): Focal length of the camera in the horizontal direction.
        fy (float): Focal length of the camera in the vertical direction.
        cx (float): Optical center of the camera in the horizontal direction.
        cy (float): Optical center of the camera in the vertical direction.
        original_height (int, optional): Height of the original depth map for grid data.
        original_width (int, optional): Width of the original depth map for grid data.
        is_grid_data (bool): If True, process depth_data as compressed grid.
        grid_rows (int, optional): Number of rows for grid data.
        grid_cols (int, optional): Number of columns for grid data.

    Returns:
        numpy.ndarray: Point cloud data with shape (N, 3). Each point is [x, y, z].
                       Returns an empty array if there are no valid points.
    """
    try:
        # Debug output
        logger.debug(f"[PointCloud] Input depth_data shape: {depth_data.shape}, range: {np.min(depth_data):.4f} to {np.max(depth_data):.4f}")
        logger.debug(f"[PointCloud] Camera params: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        if is_grid_data:
            logger.debug(f"[PointCloud] Grid mode with rows={grid_rows}, cols={grid_cols}, original size={original_height}x{original_width}")
        else:
            logger.debug("[PointCloud] Full resolution mode")
        
        # Input validation
        if depth_data is None or depth_data.size == 0:
            logger.warning("[PointCloud] Error: Empty depth data")
            return np.empty((0, 3), dtype=np.float32)
        
        if is_grid_data:
            # Grid data parameter validation
            if grid_rows is None or grid_cols is None:
                # Infer from actual data shape
                grid_rows, grid_cols = depth_data.shape[:2]
                logger.debug(f"[PointCloud] Using actual grid dimensions: {grid_rows}x{grid_cols}")
            
            # Directly generate point cloud from compressed data (vectorized for speed)
            # Create grid indices
            r_indices, c_indices = np.indices((grid_rows, grid_cols))
            
            # Valid depth value determination
            valid_mask = depth_data > 0.01
            valid_depth = depth_data[valid_mask]
            
            if valid_depth.size == 0:
                logger.warning("[PointCloud] No valid depth values in grid data")
                return np.empty((0, 3), dtype=np.float32)
                
            valid_r = r_indices[valid_mask]
            valid_c = c_indices[valid_mask]
            
            # Calculate center pixel coordinates corresponding to grid cells
            u_centers = (valid_c + 0.5)  # Center coordinates on the grid
            v_centers = (valid_r + 0.5)
            
            # Calculate 3D coordinates in bulk
            # Note: cx, cy use the center point of the grid
            x_values = (u_centers - cx) * valid_depth / fx
            y_values = (v_centers - cy) * valid_depth / fy
            z_values = valid_depth
            
            # Filter out invalid values
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
            
            # Debug output for compressed data only
            logger.debug(f"[PointCloud] Compressed grid stats - X: min={np.min(x_values):.2f}, max={np.max(x_values):.2f}")
            logger.debug(f"[PointCloud] Compressed grid stats - Y: min={np.min(y_values):.2f}, max={np.max(y_values):.2f}")
            logger.debug(f"[PointCloud] Compressed grid stats - Z: min={np.min(z_values):.2f}, max={np.max(z_values):.2f}")
            
            # Stack results
            points = np.stack((x_values, y_values, z_values), axis=-1)
            logger.info(f"[PointCloud] Generated {points.shape[0]} points from compressed grid")
            return points
            
        else:
            # Full resolution depth map processing (vectorized)
            h, w = depth_data.shape[:2]
            v_coords, u_coords = np.indices((h, w))
            
            valid_mask = depth_data > 0.01  # Target only valid depth points
            z_values = depth_data[valid_mask]
            
            if z_values.size == 0:
                logger.warning("[PointCloud] No valid depth values found")
                return np.empty((0, 3), dtype=np.float32)
            
            u_values = u_coords[valid_mask]
            v_values = v_coords[valid_mask]
            
            # Calculate 3D coordinates
            x_cam = (u_values - cx) * z_values / fx
            y_cam = (v_values - cy) * z_values / fy
            
            # Outlier filtering
            valid_idx = ~(np.isnan(x_cam) | np.isnan(y_cam) | np.isnan(z_values) | 
                          np.isinf(x_cam) | np.isinf(y_cam) | np.isinf(z_values))
            valid_idx &= (np.abs(x_cam) < 10) & (np.abs(y_cam) < 10) & (z_values < 20) & (z_values > 0)
            
            x_cam = x_cam[valid_idx]
            y_cam = y_cam[valid_idx]
            z_values = z_values[valid_idx]
            
            logger.info(f"[PointCloud] Full res mode: {valid_idx.sum()}/{valid_mask.sum()} points after filtering")
            
            # Stack results
            points = np.stack((x_cam, y_cam, z_values), axis=-1)
            return points
        
    except Exception as e:
        logger.error(f"[PointCloud] Error in depth_to_point_cloud: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
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
                Maximum height (Y coordinate) to be considered
    
    Returns:
        numpy.ndarray: Occupancy grid with shape (grid_height, grid_width)
            0: Unknown (no data)
            1: Occupied (obstacle)
            2: Free to pass
    """
    try:
        # Extract necessary parameters from grid_params
        x_min, x_max = grid_params["x_range"]
        z_min, z_max = grid_params["z_range"]
        grid_resolution = grid_params["resolution"]  # Correctly get the value of resolution
        
        # Initialization: set all cells to "unknown"
        grid_height = int((z_max - z_min) / grid_resolution)
        grid_width = int((x_max - x_min) / grid_resolution)
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        logger.info(f"[OccGrid] Creating occupancy grid: resolution={grid_resolution}m, size={grid_width}x{grid_height} cells")
        
        # Check for empty point cloud
        if points is None or not isinstance(points, np.ndarray) or points.size == 0:
            logger.warning("[OccGrid] Empty point cloud, returning default grid")
            return grid
        
        logger.debug(f"[OccGrid] Processing {points.shape[0]} points")
        logger.debug(f"[OccGrid] Point cloud data type: {points.dtype}")
        
        # Calculate the center position of the grid (based on camera position)
        grid_center_x = grid_width // 2
        grid_center_y = grid_height - 10  # Centered slightly in front of the camera
        
        # Convert point cloud data to grid coordinates (vectorized processing)
        # Map X-axis (left-right) to grid's horizontal direction
        grid_x = np.round(points[:, 0] / grid_resolution + grid_center_x).astype(int)
        # Map Z-axis (front-back) to grid's vertical direction
        grid_y = grid_center_y - np.round(points[:, 2] / grid_resolution).astype(int)
        # Use Y-axis (up-down) as height
        height = points[:, 1]
        
        # Check grid range before processing
        logger.debug(f"[OccGrid] Grid X range: {np.min(grid_x)} to {np.max(grid_x)}, Grid Y range: {np.min(grid_y)} to {np.max(grid_y)}")
        logger.debug(f"[OccGrid] Height range: {np.min(height)} to {np.max(height)}")
        
        # Check height distribution (important for floor and ceiling detection)
        height_percentiles = np.percentile(height, [5, 25, 50, 75, 95])
        logger.debug(f"[OccGrid] Height percentiles [5,25,50,75,95]: {height_percentiles}")
        
        # Adaptively determine thresholds from height statistics
        # Use 5th percentile for floor detection, 75th percentile for obstacle detection
        adaptive_floor_threshold = height_percentiles[0] * 0.7  # 70% of 5th percentile as floor threshold
        adaptive_obstacle_threshold = height_percentiles[3] * 0.5  # 50% of 75th percentile as obstacle threshold
        
        logger.info(f"[OccGrid] Using adaptive thresholds - floor: {adaptive_floor_threshold:.3f}m, obstacle: {adaptive_obstacle_threshold:.3f}m")
        
        # Process only points within the grid
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
        
        # Option: Use NumPy vectorized processing for faster execution
        # Determine the best classification for each grid cell
        
        # Optimized for compressed data: process at the grid cell level, not point level
        # Group coordinates of each cell and their height values
        unique_cells = {}  # (x, y) -> [heights]
        
        # Create mapping of grid cells to height values
        for i, (x, y, h) in enumerate(zip(grid_x, grid_y, height)):
            cell_key = (x, y)
            if cell_key not in unique_cells:
                unique_cells[cell_key] = []
            unique_cells[cell_key].append(h)
        
        # Determine classification for each cell
        logger.debug(f"[OccGrid] Processing {len(unique_cells)} unique grid cells")
        
        for (x, y), heights in unique_cells.items():
            # Calculate statistics if there are multiple height values
            heights_array = np.array(heights)
            min_height = np.min(heights_array)
            max_height = np.max(heights_array)
            
            # Classification using height thresholds (optimized for compressed data)
            # Use adaptively determined thresholds
            
            # Calculate median and standard deviation of heights (robust to noise)
            median_height = np.median(heights_array)
            height_std = np.std(heights_array)
            
            # Floor detection (points at low positions) - using adaptive threshold
            if min_height < adaptive_floor_threshold:
                # Floor (cell containing floor points) = Free to pass
                grid[y, x] = 2
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as FREE (floor): heights [{min_height:.2f} to {max_height:.2f}]")
            # If there is a large variation in height, classify as obstacle (uneven surface)
            elif height_std > abs(adaptive_floor_threshold) * 0.5:
                grid[y, x] = 1
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as OBSTACLE (high variance: {height_std:.3f}): heights [{min_height:.2f} to {max_height:.2f}]")
            # If the median height is within the threshold range, classify as floor
            elif median_height < adaptive_obstacle_threshold and median_height > adaptive_floor_threshold * 1.5:
                grid[y, x] = 2
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as FREE (median height): heights [{min_height:.2f} to {max_height:.2f}]")
            # If points are concentrated at high positions, classify as free to pass (high object)
            elif max_height > adaptive_obstacle_threshold * 3:
                grid[y, x] = 2
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as FREE (high object): heights [{min_height:.2f} to {max_height:.2f}]")
            else:
                # Otherwise, classify as obstacle
                grid[y, x] = 1
                logger.debug(f"[OccGrid] Cell ({x},{y}) classified as OBSTACLE (default): heights [{min_height:.2f} to {max_height:.2f}]")
        
        # Simple grid statistics
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
    Visualize the occupancy grid. Optimized for compressed data.
    
    Args:
        occupancy_grid: Occupancy grid (0=unknown, 1=obstacle, 2=free to pass)
        scale_factor: Factor to enlarge the display
    
    Returns:
        Visualized image
    """
    try:
        logger.info(f"[OccVis] Visualizing occupancy grid with shape {occupancy_grid.shape}, scale={scale_factor}")
        
        # Grid check
        if occupancy_grid is None or not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.size == 0:
            logger.warning("[OccVis] Invalid occupancy grid")
            return np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Output grid statistics
        unique_values, counts = np.unique(occupancy_grid, return_counts=True)
        stats = {val: count for val, count in zip(unique_values, counts)}
        logger.debug(f"[OccVis] Occupancy grid stats: {stats}")
        
        # Grid size
        grid_h, grid_w = occupancy_grid.shape
        
        # Use a larger scale factor for small grids
        if grid_h < 50 or grid_w < 50:
            logger.debug(f"[OccVis] Small grid detected, using larger scale factor: {scale_factor}")
        
        scaled_h = grid_h * scale_factor
        scaled_w = grid_w * scale_factor
        
        # Create a canvas for display (RGB)
        visualization = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
        
        # Define colors (BGR) - Enhance contrast for visibility
        colors = {
            0: [50, 50, 50],     # Unknown area: dark gray
            1: [0, 0, 255],      # Obstacle: brighter red
            2: [0, 255, 0]       # Free to pass: brighter green
        }
        
        # Draw the content of the grid (vectorized for speed)
        # Create the image all at once with NumPy operations
        for cell_value, color in colors.items():
            mask = occupancy_grid == cell_value
            if np.any(mask):
                # Expand the mask
                expanded_mask = np.repeat(np.repeat(mask, scale_factor, axis=0), scale_factor, axis=1)
                
                # Apply the color
                for c_idx, c_val in enumerate(color):
                    visualization[:, :, c_idx][expanded_mask] = c_val
        
        # Draw a point to indicate the vehicle position in the center
        # Center position in the original grid (calculated for compressed grid)
        orig_center_x, orig_center_y = grid_w // 2, grid_h - grid_h // 10
        
        # Center position after scaling (adjusted to be the center of the cell)
        center_x = orig_center_x * scale_factor + scale_factor // 2
        center_y = orig_center_y * scale_factor + scale_factor // 2
        
        # Vehicle position marker
        marker_radius = max(3, scale_factor)
        cv2.circle(visualization, (center_x, center_y), marker_radius, [255, 255, 255], -1)
        
        # Direction arrow
        arrow_length = max(10, scale_factor * 2)
        arrow_thickness = max(1, scale_factor // 2)
        cv2.arrowedLine(visualization,
                      (center_x, center_y),
                      (center_x, center_y - arrow_length),
                      [255, 255, 255],
                      arrow_thickness,
                      tipLength=0.3)
        
        # Draw grid lines (for visual reference)
        line_color = [50, 50, 50]  # Darker gray
        line_thickness = 1
        
        # Adjust interval for small grids
        grid_spacing = max(1, min(5, grid_h // 5))
        
        # Draw horizontal and vertical lines
        for i in range(0, grid_h + 1, grid_spacing):
            y = i * scale_factor
            cv2.line(visualization, (0, y), (scaled_w, y), line_color, line_thickness)
            
        for j in range(0, grid_w + 1, grid_spacing):
            x = j * scale_factor
            cv2.line(visualization, (x, 0), (x, scaled_h), line_color, line_thickness)
        
        # Display 1-meter scale
        meter_text = "1m"
        # Calculate pixels per meter from grid resolution
        grid_resolution = 0.1 * 20  # Resolution for compressed grid
        pixels_per_meter = scale_factor / grid_resolution
        meter_line_length = int(pixels_per_meter)
        
        # Draw scale bar
        scale_bar_y = scaled_h - 30
        scale_bar_x = 20
        cv2.line(visualization, 
                (scale_bar_x, scale_bar_y), 
                (scale_bar_x + meter_line_length, scale_bar_y), 
                [200, 200, 200], 2)
        # Draw scale text
        cv2.putText(visualization, meter_text, 
                   (scale_bar_x + meter_line_length // 2 - 10, scale_bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [200, 200, 200], 1)
        
        # Add cell statistics
        unknown_cells = np.sum(occupancy_grid == 0)
        obstacle_cells = np.sum(occupancy_grid == 1)
        free_cells = np.sum(occupancy_grid == 2)
        total_cells = grid_h * grid_w
        
        # Use English text utility (instead of Japanese text)
        try:
            from english_text_utils import cv2_put_english_text
            # Free area
            stats_text = f"Free: {free_cells}/{total_cells} ({free_cells/total_cells*100:.0f}%)"
            visualization = cv2_put_english_text(visualization, stats_text, (10, 20), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, [30, 220, 30], 2)
            
            # Obstacle area
            stats_text = f"Obstacle: {obstacle_cells}/{total_cells} ({obstacle_cells/total_cells*100:.0f}%)"
            visualization = cv2_put_english_text(visualization, stats_text, (10, 45), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 50, 220], 2)
        except ImportError:
            # If fix_text_encoding module is not available, use English labels
            logger.debug("fix_text_encoding module not found. Using English labels.")
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
    test_points = depth_to_point_cloud(test_depth, 500, 500)
    
    # Convert to occupancy grid
    test_grid = create_top_down_occupancy_grid(test_points, 0.05, 200, 200, 0.5)
    
    # Visualize
    test_vis = visualize_occupancy_grid(test_grid)
    
    # Save image (if needed)
    # import cv2
    # cv2.imwrite("test_topview.jpg", test_vis)
    
    print("Test completed successfully")