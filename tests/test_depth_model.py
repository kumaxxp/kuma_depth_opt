import pytest
import numpy as np
from depth_processor.depth_model import DepthProcessor, convert_to_absolute_depth, HAS_AXENGINE

# To run tests, navigate to the root of your project (c:\work\kuma_depth_opt)
# and run: pytest

# --- Tests for DepthProcessor Class ---

def test_depth_processor_initialization(dummy_config_linux, monkeypatch):
    """Test DepthProcessor initialization."""
    # Ensure HAS_AXENGINE is False for this test to prevent actual model loading attempts
    monkeypatch.setattr("depth_processor.depth_model.HAS_AXENGINE", False)
    processor = DepthProcessor(dummy_config_linux)
    assert processor.config == dummy_config_linux
    assert processor.model_path == dummy_config_linux["depth_model"]["model_path"]
    assert processor.model is None # Because HAS_AXENGINE is False
    assert processor.input_name is None
    assert processor.model_input_width == dummy_config_linux["depth_model_parameters"]["input_width"]
    assert processor.model_input_height == dummy_config_linux["depth_model_parameters"]["input_height"]

def test_depth_processor_process_frame(dummy_depth_processor):
    """Test frame preprocessing."""
    # Create a dummy frame (e.g., 640x480x3)
    # The actual content doesn't matter much for this test, only dimensions and type
    original_height, original_width = 480, 640
    dummy_frame = np.random.randint(0, 256, (original_height, original_width, 3), dtype=np.uint8)
    
    processed_tensor = dummy_depth_processor.process_frame(dummy_frame)
    
    expected_height = dummy_depth_processor.model_input_height
    expected_width = dummy_depth_processor.model_input_width
    
    assert processed_tensor.shape == (1, expected_height, expected_width, 3)
    assert processed_tensor.dtype == np.uint8 # or the type it's converted to if any

def test_depth_processor_predict_dummy_data(dummy_depth_processor, dummy_config_linux):
    """Test predict method when model is not available (should return dummy data)."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Ensure model is None to force dummy data generation
    dummy_depth_processor.model = None
    dummy_depth_processor.input_name = None
    
    depth_map, inference_time = dummy_depth_processor.predict(dummy_frame)
    
    expected_height = dummy_config_linux["depth_model_parameters"]["input_height"]
    expected_width = dummy_config_linux["depth_model_parameters"]["input_width"]
    
    assert depth_map.shape == (1, expected_height, expected_width, 1)
    assert depth_map.dtype == np.float32
    assert inference_time >= 0
    # Check if dummy data has the expected gradient pattern (optional, but good)
    assert np.all(depth_map[0, 0, :, 0] < depth_map[0, -1, :, 0]) # Simple check for y-gradient

# --- Tests for compress_depth_to_grid Method ---

def test_compress_depth_to_grid_normal_case_mean(dummy_depth_processor, dummy_config_linux):
    """Test grid compression with mean method."""
    model_h = dummy_config_linux["depth_model_parameters"]["input_height"]
    model_w = dummy_config_linux["depth_model_parameters"]["input_width"]
    
    # Create a simple gradient depth map for testing
    depth_map_abs = np.array([[ (r+c) for c in range(model_w)] for r in range(model_h)], dtype=np.float32)
    
    grid_config = dummy_config_linux["grid_compressor"]
    grid_config["method"] = "mean"
    
    compressed_grid = dummy_depth_processor.compress_depth_to_grid(depth_map_abs, grid_config)
    
    assert compressed_grid is not None
    assert compressed_grid.shape == (grid_config["target_rows"], grid_config["target_cols"])
    # Add more specific value checks if possible, e.g., for a known simple input
    # For example, if input is all 1s, output should be all 1s.
    # If input is a ramp, calculate expected means for a few cells.

def test_compress_depth_to_grid_empty_input(dummy_depth_processor, dummy_config_linux):
    """Test grid compression with empty input."""
    empty_map = np.array([], dtype=np.float32)
    compressed_grid = dummy_depth_processor.compress_depth_to_grid(empty_map, dummy_config_linux["grid_compressor"])
    assert compressed_grid is None

    none_map = None
    compressed_grid_none = dummy_depth_processor.compress_depth_to_grid(none_map, dummy_config_linux["grid_compressor"])
    assert compressed_grid_none is None

def test_compress_depth_to_grid_small_input_causes_zero_cell_dim(dummy_depth_processor, dummy_config_linux):
    """Test grid compression where input is smaller than target grid, causing zero cell height/width."""
    # Input map smaller than target grid cells
    depth_map_abs = np.ones((dummy_config_linux["grid_compressor"]["target_rows"] -1, 
                               dummy_config_linux["grid_compressor"]["target_cols"]-1), dtype=np.float32)
    grid_config = dummy_config_linux["grid_compressor"]
    compressed_grid = dummy_depth_processor.compress_depth_to_grid(depth_map_abs, grid_config)
    assert compressed_grid is None # Expect None due to zero cell height/width


# --- Tests for convert_to_absolute_depth Function ---

@pytest.mark.parametrize("is_compressed", [False, True])
def test_convert_to_absolute_depth_normal_case(dummy_config_linux, is_compressed):
    """Test absolute depth conversion for full and compressed maps."""
    config = dummy_config_linux
    model_h = config["depth_model_parameters"]["input_height"]
    model_w = config["depth_model_parameters"]["input_width"]
    grid_rows = config["grid_compressor"]["target_rows"]
    grid_cols = config["grid_compressor"]["target_cols"]

    if is_compressed:
        # Simulate a compressed relative depth map (e.g., 0.1 to 0.9 range)
        relative_depth_map = np.random.rand(grid_rows, grid_cols).astype(np.float32) * 0.8 + 0.1
    else:
        # Simulate a full relative depth map (e.g., 0.1 to 0.9 range)
        relative_depth_map = np.random.rand(model_h, model_w).astype(np.float32) * 0.8 + 0.1
        
    absolute_depth = convert_to_absolute_depth(relative_depth_map, config, is_compressed_grid=is_compressed)
    
    assert absolute_depth.shape == relative_depth_map.shape
    assert absolute_depth.dtype == np.float32
    # Check that values are within a plausible range based on config (e.g., min_depth_m, max_depth_m)
    # This requires careful calculation based on the conversion logic
    # For a simple check, ensure no NaNs and values are positive
    assert not np.isnan(absolute_depth).any()
    assert np.all(absolute_depth >= 0) 
    # A more specific check based on the transformation:
    # Values are scaled from (effective_far, effective_near) to (0.5, 0.5 + scaling_factor)
    # then clipped. For input 0.1 to 0.9, and typical scaling, most values should be > 0.5.
    min_expected_val = 0.1 # from final clipping
    max_expected_val = config["depth_processing"].get("max_depth_m", 50.0) # from final clipping
    assert np.all(absolute_depth >= min_expected_val)
    assert np.all(absolute_depth <= max_expected_val) 

def test_convert_to_absolute_depth_empty_input(dummy_config_linux):
    """Test absolute depth conversion with empty input."""
    empty_map = np.array([], dtype=np.float32)
    config = dummy_config_linux
    abs_depth_empty = convert_to_absolute_depth(empty_map, config, is_compressed_grid=False)
    
    # Determine expected fallback shape
    model_params_conf = config.get("depth_model_parameters", { })
    fb_h = model_params_conf.get("input_height", 256)
    fb_w = model_params_conf.get("input_width", 384)
    expected_fallback_shape = (fb_h, fb_w)

    assert abs_depth_empty.shape == expected_fallback_shape
    assert np.all(abs_depth_empty == 3.0) # Default fallback value

    abs_depth_none = convert_to_absolute_depth(None, config, is_compressed_grid=False)
    assert abs_depth_none.shape == expected_fallback_shape
    assert np.all(abs_depth_none == 3.0)

def test_convert_to_absolute_depth_all_invalid_input(dummy_config_linux):
    """Test with input where all values are <= 0.01 (considered invalid)."""
    config = dummy_config_linux
    model_h = config["depth_model_parameters"]["input_height"]
    model_w = config["depth_model_parameters"]["input_width"]
    # All values are 0.005, which is < 0.01
    invalid_map = np.full((model_h, model_w), 0.005, dtype=np.float32)
    
    absolute_depth = convert_to_absolute_depth(invalid_map, config, is_compressed_grid=False)
    assert absolute_depth.shape == invalid_map.shape
    assert np.all(absolute_depth == 3.0) # Should return default depth

