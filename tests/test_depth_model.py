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
    min_expected_val = config["depth_processing"].get("min_depth_m", 0.1) # from final clipping
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

@pytest.mark.parametrize("is_compressed", [False, True])
@pytest.mark.parametrize("invalid_value", [0.0, 0.005, -0.1, np.nan, np.inf])
def test_convert_to_absolute_depth_various_invalid_inputs(dummy_config_linux, is_compressed, invalid_value):
    """Test convert_to_absolute_depth with various types of invalid single value inputs."""
    config = dummy_config_linux
    model_h = config["depth_model_parameters"]["input_height"]
    model_w = config["depth_model_parameters"]["input_width"]
    grid_rows = config["grid_compressor"]["target_rows"]
    grid_cols = config["grid_compressor"]["target_cols"]

    if is_compressed:
        input_shape = (grid_rows, grid_cols)
    else:
        input_shape = (model_h, model_w)

    # Create an input map filled with the specific invalid value
    # Handle np.nan and np.inf separately for creation if necessary
    if np.isnan(invalid_value):
        invalid_map = np.full(input_shape, np.nan, dtype=np.float32)
    elif np.isinf(invalid_value):
        invalid_map = np.full(input_shape, np.inf, dtype=np.float32)
    else:
        invalid_map = np.full(input_shape, invalid_value, dtype=np.float32)

    absolute_depth = convert_to_absolute_depth(invalid_map, config, is_compressed_grid=is_compressed)
    
    assert absolute_depth.shape == input_shape
    # Expect fallback to default depth (3.0) or a clipped value if config is extreme
    # For most invalid inputs, it should be the default_depth_fallback_value (3.0)
    # If the invalid_value was np.inf, it might get clipped by max_depth_m if logic changes.
    # Current logic should result in 3.0 for all these invalid inputs as they won't pass `depth_map > 0.01`
    # or will be handled by NaN/Inf specific checks if any were added to the main function.
    # Based on current convert_to_absolute_depth, it should return default 3.0.
    assert np.all(absolute_depth == 3.0)


@pytest.mark.parametrize("is_compressed", [False, True])
def test_convert_to_absolute_depth_extreme_config_values(dummy_config_linux, is_compressed, monkeypatch):
    """Test convert_to_absolute_depth with extreme configuration values."""
    config = dummy_config_linux
    
    # Modify config for extreme values
    extreme_depth_processing_config = {
        "scaling_factor": 1000.0, # Very large scaling
        "depth_scale": 10.0,      # Very large scale for non-compressed
        "min_depth_m": 10.0,      # High min depth
        "max_depth_m": 5.0        # Low max depth (min > max)
    }
    monkeypatch.setitem(config, "depth_processing", extreme_depth_processing_config)

    model_h = config["depth_model_parameters"]["input_height"]
    model_w = config["depth_model_parameters"]["input_width"]
    grid_rows = config["grid_compressor"]["target_rows"]
    grid_cols = config["grid_compressor"]["target_cols"]

    if is_compressed:
        relative_depth_map = np.random.rand(grid_rows, grid_cols).astype(np.float32) * 0.8 + 0.1
    else:
        relative_depth_map = np.random.rand(model_h, model_w).astype(np.float32) * 0.8 + 0.1
        
    absolute_depth = convert_to_absolute_depth(relative_depth_map, config, is_compressed_grid=is_compressed)
    
    assert absolute_depth.shape == relative_depth_map.shape
    # With min_depth_m = 10.0 and max_depth_m = 5.0, all values should be clipped to max_depth_m (5.0)
    # because the clipping logic is `np.clip(absolute_depth, min_clip, max_clip)`
    # if min_clip > max_clip, it effectively clips all values to max_clip if they are below,
    # or to min_clip if they are above. Given the calculation, values will likely be positive.
    # The final np.clip(..., 10.0, 5.0) will make all values 5.0.
    assert np.all(absolute_depth == extreme_depth_processing_config["max_depth_m"])

    # Test with min_depth_m < max_depth_m but still extreme
    extreme_depth_processing_config_2 = {
        "scaling_factor": 0.001, # Very small scaling
        "depth_scale": 0.01,
        "min_depth_m": 0.01,
        "max_depth_m": 100.0
    }
    monkeypatch.setitem(config, "depth_processing", extreme_depth_processing_config_2)
    absolute_depth_2 = convert_to_absolute_depth(relative_depth_map, config, is_compressed_grid=is_compressed)
    assert np.all(absolute_depth_2 >= extreme_depth_processing_config_2["min_depth_m"])
    assert np.all(absolute_depth_2 <= extreme_depth_processing_config_2["max_depth_m"])


def test_depth_processor_predict_model_available_mocked(dummy_config_linux, monkeypatch):
    """Test predict method when a model is available (mocked)."""
    
    # --- Mocking HAS_AXENGINE and axengine.InferenceSession ---
    monkeypatch.setattr("depth_processor.depth_model.HAS_AXENGINE", True)
    # --- ADDED: Mock os.path.exists to prevent early exit ---
    monkeypatch.setattr("depth_processor.depth_model.os.path.exists", lambda path: True)


    class MockModelInput:
        def __init__(self, name):
            self.name = name

    class MockInferenceSession:
        def __init__(self, model_path):
            self.model_path = model_path
            self.inputs = [MockModelInput("input_tensor_name")]
            self.run_called = False
            self.run_input_data = None

        def get_inputs(self):
            return self.inputs

        def run(self, output_names, input_feed):
            self.run_called = True
            self.run_input_data = input_feed
            # Simulate a model output: (1, H, W, 1)
            # Use dimensions from config for consistency
            h = dummy_config_linux["depth_model_parameters"]["input_height"]
            w = dummy_config_linux["depth_model_parameters"]["input_width"]
            # Create a dummy output similar to what the model might produce
            # (e.g., values between 0 and 1, or some other range)
            # For this test, a simple gradient like the dummy data is fine.
            dummy_output = np.zeros((1, h, w, 1), dtype=np.float32)
            for y_idx in range(h):
                value = 0.1 + 0.8 * (y_idx / h) # Normalized 0.1 to 0.9
                dummy_output[0, y_idx, :, 0] = value
            return [dummy_output]

    mock_session_instance = MockInferenceSession(dummy_config_linux["depth_model"]["model_path"])
    
    # Monkeypatch the InferenceSession constructor
    monkeypatch.setattr("depth_processor.depth_model.axe.InferenceSession", lambda path: mock_session_instance)

    # --- Test ---
    processor = DepthProcessor(dummy_config_linux)
    assert processor.is_available() # Model should now be considered available
    assert processor.model == mock_session_instance
    assert processor.input_name == "input_tensor_name"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Example camera frame
    depth_map, inference_time = processor.predict(dummy_frame)

    assert mock_session_instance.run_called
    assert "input_tensor_name" in mock_session_instance.run_input_data
    # Check input tensor properties (shape, dtype)
    input_tensor = mock_session_instance.run_input_data["input_tensor_name"]
    expected_h = dummy_config_linux["depth_model_parameters"]["input_height"]
    expected_w = dummy_config_linux["depth_model_parameters"]["input_width"]
    assert input_tensor.shape == (1, expected_h, expected_w, 3) # After process_frame
    assert input_tensor.dtype == np.uint8 # Or whatever process_frame converts to

    # Check output depth_map properties
    assert depth_map is not None
    assert depth_map.shape == (1, expected_h, expected_w, 1)
    assert depth_map.dtype == np.float32
    assert inference_time > 0 # Should be a small positive value

    # Verify the content of the depth_map (matches the mocked output)
    # This checks if the output from the mocked model.run is correctly returned
    assert np.all(depth_map[0,0,:,0] < depth_map[0,-1,:,0]) # Check y-gradient from mock output


def test_compress_depth_to_grid_with_nans(dummy_depth_processor, dummy_config_linux):
    """Test grid compression when input depth map contains NaNs."""
    model_h = dummy_config_linux["depth_model_parameters"]["input_height"]
    model_w = dummy_config_linux["depth_model_parameters"]["input_width"]
    
    # Create a depth map with some NaNs
    depth_map_abs = np.array([[ (r+c) for c in range(model_w)] for r in range(model_h)], dtype=np.float32)
    depth_map_abs[model_h//4, model_w//4] = np.nan # Introduce a NaN
    depth_map_abs[model_h//2, model_w//2] = np.nan # Introduce another NaN

    grid_config = dummy_config_linux["grid_compressor"].copy() # Use .copy() to avoid modifying fixture
    grid_config["method"] = "mean" # Explicitly set to mean for np.nanmean behavior
    
    # Ensure the logger is accessible or mock it if it causes issues during test
    # from depth_processor import depth_model
    # monkeypatch.setattr(depth_model, "logger", MagicMock())


    compressed_grid = dummy_depth_processor.compress_depth_to_grid(depth_map_abs, grid_config)
    
    assert compressed_grid is not None
    assert compressed_grid.shape == (grid_config["target_rows"], grid_config["target_cols"])
    
    # The current implementation of compress_depth_to_grid uses np.mean, which propagates NaNs.
    # If a cell's ROI contains only NaNs, or if np.mean is used and any value is NaN, the result is NaN.
    # The code has a specific check: `if np.isnan(value): compressed_grid[r, c] = 0.0`
    # So, we expect NaNs to be converted to 0.0 in the output.
    assert not np.isnan(compressed_grid).any()

    # For a more robust test, we could calculate an expected value for a cell
    # known to contain a NaN and one known not to.
    # For now, just checking no NaNs in output is a good first step given the handling.
    # If a cell's ROI becomes all NaNs, np.mean would be NaN, then converted to 0.0.
    # If a cell's ROI has some NaNs and some numbers, np.mean is NaN, then converted to 0.0.
    # This means cells affected by NaNs will become 0.0.

    # Create a map that is ALL NaNs
    all_nan_map = np.full((model_h, model_w), np.nan, dtype=np.float32)
    compressed_all_nan_grid = dummy_depth_processor.compress_depth_to_grid(all_nan_map, grid_config)
    assert compressed_all_nan_grid is not None
    assert np.all(compressed_all_nan_grid == 0.0) # All cells should be 0.0

# --- Fixture for DepthProcessor with mocked model (if needed for more tests) ---
# Consider adding a fixture that provides a DepthProcessor with a mocked model
# if multiple tests need this setup.
# For now, the test_depth_processor_predict_model_available_mocked handles its own mocking.

# Ensure all existing tests still pass with these additions.
# (No changes to existing tests, only additions)

