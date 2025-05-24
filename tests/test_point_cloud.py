import numpy as np
import pytest
from depth_processor.point_cloud import depth_to_point_cloud, create_top_down_occupancy_grid
from tests.conftest import dummy_config_linux # Import the fixture

@pytest.fixture
def sample_depth_data():
    return np.array([
        [1.0, 1.5, 0.0],
        [2.0, 2.5, 3.0],
        [0.0, 3.5, 4.0]
    ], dtype=np.float32)

@pytest.fixture
def sample_camera_intrinsics():
    return {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

@pytest.fixture
def sample_grid_config():
    return {"target_rows": 3, "target_cols": 3} # Matches sample_depth_data shape

@pytest.fixture
def sample_original_image_dims():
    return (480, 640) # Example original dimensions

def test_depth_to_point_cloud_full_res(sample_depth_data, sample_camera_intrinsics):
    """Test point cloud generation from full resolution depth map."""
    points = depth_to_point_cloud(
        sample_depth_data,
        camera_intrinsics=sample_camera_intrinsics,
        is_grid_data=False
    )
    # Expect points for non-zero depth values
    # (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0) -> 7 valid points
    assert points.shape[0] == np.count_nonzero(sample_depth_data > 0.01) 
    assert points.shape[1] == 3

def test_depth_to_point_cloud_grid_data(sample_depth_data, sample_camera_intrinsics, sample_grid_config, sample_original_image_dims):
    """Test point cloud generation from grid data."""
    points = depth_to_point_cloud(
        sample_depth_data,
        camera_intrinsics=sample_camera_intrinsics,
        is_grid_data=True,
        grid_config=sample_grid_config,
        original_image_dims=sample_original_image_dims
    )
    assert points.shape[0] == np.count_nonzero(sample_depth_data > 0.01)
    assert points.shape[1] == 3

def test_depth_to_point_cloud_empty_input(sample_camera_intrinsics):
    """Test with empty depth data."""
    empty_depth = np.array([[]], dtype=np.float32)
    points = depth_to_point_cloud(
        empty_depth,
        camera_intrinsics=sample_camera_intrinsics,
        is_grid_data=False
    )
    assert points.shape == (0, 3)

def test_depth_to_point_cloud_missing_intrinsics(sample_depth_data):
    """Test with missing camera intrinsics."""
    points = depth_to_point_cloud(
        sample_depth_data,
        camera_intrinsics={"fx": 500.0}, # Missing fy, cx, cy
        is_grid_data=False
    )
    assert points.shape == (0,3)


@pytest.fixture
def sample_points_for_occupancy():
    # Points representing a small cluster
    return np.array([
        [0.5, 0.1, 1.0],  # x, y, z (depth)
        [0.6, 0.15, 1.1],
        [-0.5, 0.2, 1.5], # Should be in a different cell
        [0.55, 0.8, 1.05], # Higher, should still be an obstacle
        [0.1, 0.05, 2.5] # Further away, free space if y is low
    ], dtype=np.float32)

@pytest.fixture
def sample_grid_params():
    return {
        "x_range": (-1.0, 1.0),
        "z_range": (0.0, 3.0), # Adjusted to cover z up to 2.5
        "resolution": 0.5,
        "y_min_filter": -0.1, # Consider points slightly below ground
        "y_max_filter": 1.0,  # Max height to consider for points
        "obstacle_height_threshold": 0.1, # Min height of point to be obstacle
        "free_space_max_height": 0.08 # Max height of point to be considered free space
    }

def test_create_top_down_occupancy_grid_basic(sample_points_for_occupancy, sample_grid_params):
    """Test basic occupancy grid creation."""
    grid = create_top_down_occupancy_grid(sample_points_for_occupancy, sample_grid_params)
    
    # Expected grid dimensions:
    # z_range_len = 3.0 - 0.0 = 3.0 -> 3.0 / 0.5 = 6 cells for height (z-axis)
    # x_range_len = 1.0 - (-1.0) = 2.0 -> 2.0 / 0.5 = 4 cells for width (x-axis)
    assert grid.shape == (6, 4) # (grid_height_cells, grid_width_cells)
    
    # Check a few expected occupied cells based on sample_points_for_occupancy and sample_grid_params
    # Point [0.5, 0.1, 1.0]:
    # grid_x = int((0.5 - (-1.0)) / 0.5) = int(1.5 / 0.5) = int(3) = 3
    # grid_z = int((1.0 - 0.0) / 0.5) = int(1.0 / 0.5) = int(2) = 2
    # Y = 0.1, obstacle_height_threshold = 0.1. So, points with Y >= 0.1 are obstacles.
    # This point is an obstacle.
    assert grid[2, 3] == 1 # Occupied (obstacle)

    # Point [-0.5, 0.2, 1.5]:
    # grid_x = int((-0.5 - (-1.0)) / 0.5) = int(0.5 / 0.5) = int(1) = 1
    # grid_z = int((1.5 - 0.0) / 0.5) = int(1.5 / 0.5) = int(3) = 3
    # Y = 0.2 >= 0.1 (obstacle_height_threshold) -> obstacle
    assert grid[3, 1] == 1 # Occupied

    # Point [0.1, 0.05, 2.5]:
    # grid_x = int((0.1 - (-1.0)) / 0.5) = int(1.1 / 0.5) = int(2.2) = 2
    # grid_z = int((2.5 - 0.0) / 0.5) = int(2.5 / 0.5) = int(5) = 5
    # Y = 0.05 < 0.08 (free_space_max_height) -> free space
    assert grid[5, 2] == 2 # Free space

def test_create_top_down_occupancy_grid_empty_points(sample_grid_params):
    """Test with empty point cloud."""
    empty_points = np.empty((0, 3), dtype=np.float32)
    grid = create_top_down_occupancy_grid(empty_points, sample_grid_params)
    assert grid.shape == (6, 4) # Should still create an empty grid of correct size
    assert np.all(grid == 0)     # All cells should be unknown

def test_create_top_down_occupancy_grid_points_outside_y_filter(sample_grid_params):
    """Test points filtered out by y_min_filter and y_max_filter."""
    points = np.array([
        [0.5, 1.5, 1.0],  # Y > y_max_filter (1.0)
        [0.6, -0.5, 1.1] # Y < y_min_filter (-0.1)
    ], dtype=np.float32)
    grid = create_top_down_occupancy_grid(points, sample_grid_params)
    assert grid.shape == (6,4)
    assert np.all(grid == 0) # All cells should be unknown as points are filtered out
