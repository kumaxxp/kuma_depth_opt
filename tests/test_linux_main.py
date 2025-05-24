\
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Add the project root to the path to allow importing linux_main
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import app from linux_main AFTER potentially patching dependencies or setting up mocks
# This is tricky because linux_main.py runs initialization code at import time (logger setup)
# and FastAPI startup events handle the rest.
# For now, let's assume direct import and see.
from linux_main import app, PointCloudResponse

@pytest.fixture(scope="module")
def client():
    # The TestClient handles startup and shutdown events by default
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_config_data():
    return {
        "logging": {"level": "DEBUG"},
        "depth_model": {
            "model_path": "dummy/model.axmodel"
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
        "grid_compression": {
            "enabled": False,
            "target_rows": 12,
            "target_cols": 16,
            "method": "mean"
        },
        "point_cloud": {
            "camera_intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 192.0, "cy": 128.0}
        },
        "camera": {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 10
        },
        "api_server": {
            "host": "0.0.0.0",
            "port": 8000
        }
    }

def test_get_pointcloud_system_not_initialized(client):
    # This test assumes that startup_event might fail or globals are not set.
    # We need a way to ensure globals in linux_main are None for this test.
    # One way is to patch the globals directly in the linux_main module.
    
    # Test with all components mocked to None (simulating startup failure)
    with patch("linux_main.config", None), \
         patch("linux_main.depth_processor_instance", None), \
         patch("linux_main.camera_capture", None):
        
        response = client.get("/pointcloud")
        assert response.status_code == 200 # Endpoint itself is up
        data = response.json()
        assert data["error_message"] == "System not initialized"
        assert data["point_cloud"] == []
        assert data["processing_time_total"] == 0

def test_get_pointcloud_success(client, mock_config_data):
    mock_frame = np.random.randint(0, 256, (mock_config_data["camera"]["height"], mock_config_data["camera"]["width"], 3), dtype=np.uint8)
    
    # Raw output from predict is typically 4D
    mock_raw_depth_output = np.random.rand(
        1, 
        mock_config_data["depth_model_parameters"]["input_height"], 
        mock_config_data["depth_model_parameters"]["input_width"], 
        1
    ).astype(np.float32)
    
    # This mock_depth_map is used for assertions related to later stages (e.g., input to point_cloud conversion if convert_to_absolute_depth is also mocked)
    mock_depth_map_for_conversion = np.random.rand(
        mock_config_data["depth_model_parameters"]["input_height"], 
        mock_config_data["depth_model_parameters"]["input_width"]
    ).astype(np.float32)

    mock_point_cloud_list = [[float(i), float(i), float(i)] for i in range(5)] # Simple mock
    mock_point_cloud_array = np.array(mock_point_cloud_list)

    with patch("linux_main.load_config", return_value=mock_config_data), \
         patch("linux_main.CamInput") as MockCamInput, \
         patch("linux_main.initialize_depth_model") as MockInitializeDepthModel, \
         patch("linux_main.depth_to_point_cloud", return_value=mock_point_cloud_array) as MockDepthToPointCloud, \
         patch("linux_main.convert_to_absolute_depth", return_value=mock_depth_map_for_conversion) as MockConvertToAbsoluteDepth:

        # Configure CamInput mock
        mock_cam_instance = MockCamInput.return_value
        mock_cam_instance.get_frame.return_value = mock_frame
        mock_cam_instance.width = mock_config_data["camera"]["width"]
        mock_cam_instance.height = mock_config_data["camera"]["height"]

        # Configure DepthProcessor mock (returned by initialize_depth_model)
        mock_depth_processor = MockInitializeDepthModel.return_value
        # predict returns: relative_depth_map, processed_input_image (or None)
        mock_depth_processor.predict.return_value = (mock_raw_depth_output, None) 
        # compress_depth_to_grid's return value (if called)
        mock_depth_processor.compress_depth_to_grid.return_value = mock_depth_map_for_conversion 


        # Patching globals directly within linux_main for the scope of this test
        # This ensures the endpoint uses our mocked instances.
        with patch("linux_main.config", mock_config_data), \
             patch("linux_main.camera_capture", mock_cam_instance), \
             patch("linux_main.depth_processor_instance", mock_depth_processor):

            response = client.get("/pointcloud")
            assert response.status_code == 200
            data = response.json()

            assert data["error_message"] is None
            assert data["point_cloud"] == mock_point_cloud_list
            assert "processing_time_total" in data
            assert data["processing_time_total"] > 0
            assert "processing_time_depth" in data
            assert "processing_time_compression" in data
            assert "processing_time_pointcloud" in data
            
            mock_cam_instance.get_frame.assert_called_once()
            mock_depth_processor.predict.assert_called_once_with(mock_frame)
            
            # Assert calls to convert_to_absolute_depth
            # First call is with the squeezed raw output
            squeezed_raw_output = np.squeeze(mock_raw_depth_output)
            call_args_list = MockConvertToAbsoluteDepth.call_args_list
            
            # Check the first call to convert_to_absolute_depth
            # (this creates current_depth_data)
            assert len(call_args_list) > 0, "convert_to_absolute_depth was not called"
            args, kwargs = call_args_list[0]
            assert np.array_equal(args[0], squeezed_raw_output)
            assert args[1] == mock_config_data
            assert kwargs.get("is_compressed_grid") == False

            if mock_config_data.get("grid_compression", {}).get("enabled", False):
                # If compression is enabled, convert_to_absolute_depth is called a second time
                # with the output of compress_depth_to_grid.
                assert len(call_args_list) > 1, "convert_to_absolute_depth not called for compressed grid"
                args_comp, kwargs_comp = call_args_list[1]
                assert np.array_equal(args_comp[0], mock_depth_map_for_conversion) # compress_depth_to_grid returns this
                assert args_comp[1] == mock_config_data
                assert kwargs_comp.get("is_compressed_grid") == True
                
                # compress_depth_to_grid is called with current_depth_data, which is the result of the first convert_to_absolute_depth call
                mock_depth_processor.compress_depth_to_grid.assert_called_once_with(mock_depth_map_for_conversion) 
            else:
                mock_depth_processor.compress_depth_to_grid.assert_not_called()
                assert len(call_args_list) == 1, "convert_to_absolute_depth called more than once when compression is off"


            MockDepthToPointCloud.assert_called_once()
            args_pc, kwargs_pc = MockDepthToPointCloud.call_args
            # The first argument to depth_to_point_cloud is depth_for_pointcloud_conversion
            # which is the result of the last call to convert_to_absolute_depth (i.e., mock_depth_map_for_conversion)
            assert np.array_equal(args_pc[0], mock_depth_map_for_conversion)
            assert args_pc[1] == mock_config_data["point_cloud"]["camera_intrinsics"]
            assert args_pc[2] == (mock_config_data["camera"]["height"], mock_config_data["camera"]["width"])


# More tests will be added here.
# For example, a test for a successful point cloud generation:
# def test_get_pointcloud_success(client, mock_config_data):
#     # This test requires mocking:
#     # - load_config to return mock_config_data
#     # - CamInput and its get_frame method
#     # - initialize_depth_model and its predict/compress_depth_to_grid methods
#     # - depth_to_point_cloud
#     pass

