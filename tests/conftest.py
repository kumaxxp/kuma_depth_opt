import pytest
import numpy as np

@pytest.fixture
def dummy_config_linux():
    return {
        "depth_model": {
            "model_path": "dummy/path/to/model.axmodel" # Test with a dummy path
        },
        "depth_model_parameters": {
            "input_width": 384,
            "input_height": 256
        },
        "depth_processing": {
            "scaling_factor": 10.0,
            "depth_scale": 1.0,
            "min_depth_m": 0.2,
            "max_depth_m": 10.0
        },
        "grid_compressor": {
            "enabled": True,
            "target_rows": 8,
            "target_cols": 6,
            "method": "mean" # or "max", "min"
        },
        "camera_intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 192.0,
            "cy": 128.0
        },
        "logging": {
            "level": "INFO"
        }
    }

@pytest.fixture
def dummy_depth_processor(dummy_config_linux):
    # Mock axengine if not available or for consistent testing
    try:
        from depth_processor.depth_model import DepthProcessor
    except ImportError: # Handle if path issues arise in testing environment
        # This might require adjusting PYTHONPATH or how tests are run
        # For now, assume it can be imported
        pass
        
    # Temporarily disable actual model loading for most unit tests
    # by ensuring HAS_AXENGINE is False or by further mocking
    # For now, we rely on the dummy path failing to load a real model
    # and the class handling it by not setting self.model
    
    # If axengine is truly needed for some tests, those tests would need
    # to be marked or handled specially.
    # For basic logic tests, we can often work with model=None
    
    # Ensure HAS_AXENGINE is False for tests that shouldn't load a model
    # This can be done by patching `depth_processor.depth_model.HAS_AXENGINE`
    
    processor = DepthProcessor(dummy_config_linux)
    # To ensure predict uses dummy data if model isn't (or can't be) loaded:
    processor.model = None 
    processor.input_name = None
    return processor
