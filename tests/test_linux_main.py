import pytest
import sys
import os

# Skip all tests in this file if not on Linux
if not sys.platform.startswith('linux'):
    pytestmark = pytest.mark.skip(reason="Linux-specific tests")
else:
    # All Linux-specific imports and code go here
    from fastapi.testclient import TestClient
    from unittest.mock import patch, MagicMock
    import numpy as np

    # Add the project root to the path to allow importing linux_main
    # This needs to be done before importing from linux_main
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
        
    from linux_main import app, PointCloudResponse

    @pytest.fixture # Default scope is "function"
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
            "grid_compression": { # Corrected key from "grid_compressor"
                "enabled": False,
                "target_rows": 12,
                "target_cols": 16,
                "method": "mean"
            },
            "point_cloud": { # This was "camera_intrinsics" in dummy_config_linux
                "camera_intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 192.0, "cy": 128.0}
            },
            "camera": { # This was not in dummy_config_linux but is used by linux_main
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 10
            },
            "api_server": { # This was not in dummy_config_linux
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
        # Ensure grid_compression is disabled for this specific test case
        # to match the assertion for processing_time_compression
        mock_config_data_no_compression = mock_config_data.copy()
        mock_config_data_no_compression["grid_compression"] = {
            "enabled": False, # Explicitly disable
            "target_rows": 12,
            "target_cols": 16,
            "method": "mean"
        }

        mock_frame = np.random.randint(0, 256, (mock_config_data_no_compression["camera"]["height"], mock_config_data_no_compression["camera"]["width"], 3), dtype=np.uint8)     

        # Raw output from predict is typically 4D
        mock_raw_depth_output = np.random.rand(
            1,
            mock_config_data_no_compression["depth_model_parameters"]["input_height"],
            mock_config_data_no_compression["depth_model_parameters"]["input_width"],
            1
        ).astype(np.float32)

        # This mock_depth_map is used for assertions related to later stages
        mock_depth_map_for_conversion = np.random.rand(
            mock_config_data_no_compression["depth_model_parameters"]["input_height"],
            mock_config_data_no_compression["depth_model_parameters"]["input_width"]
        ).astype(np.float32)

        mock_point_cloud_list = [[float(i), float(i), float(i)] for i in range(5)] # Simple mock
        mock_point_cloud_array = np.array(mock_point_cloud_list)

        # Mock the global config that would be loaded by lifespan manager
        with patch("linux_main.config", mock_config_data_no_compression), \
             patch("linux_main.CamInput") as MockCamInput, \
             patch("linux_main.initialize_depth_model") as MockInitializeDepthModel, \
             patch("linux_main.depth_to_point_cloud", return_value=mock_point_cloud_array) as MockDepthToPointCloud, \
             patch("linux_main.convert_to_absolute_depth", return_value=mock_depth_map_for_conversion) as MockConvertToAbsoluteDepth:

            # Configure CamInput mock
            mock_cam_instance = MockCamInput.return_value
            mock_cam_instance.get_frame.return_value = mock_frame
            mock_cam_instance.width = mock_config_data_no_compression["camera"]["width"]
            mock_cam_instance.height = mock_config_data_no_compression["camera"]["height"]

            # Configure DepthProcessor mock (returned by initialize_depth_model)
            mock_depth_processor = MockInitializeDepthModel.return_value
            mock_depth_processor.predict.return_value = (mock_raw_depth_output, None)
            # compress_depth_to_grid should not be called if compression is disabled
            # but if it were, its return value is set.
            mock_depth_processor.compress_depth_to_grid.return_value = mock_depth_map_for_conversion 

            with patch("linux_main.camera_capture", mock_cam_instance), \
                 patch("linux_main.depth_processor_instance", mock_depth_processor):

                response = client.get("/pointcloud")
                assert response.status_code == 200
                data = response.json()

                assert data["error_message"] is None
                assert data["point_cloud"] == mock_point_cloud_list
                assert "processing_time_total" in data
                assert data["processing_time_total"] > 0 
                assert "processing_time_depth" in data
                assert "processing_time_pointcloud" in data
                
                mock_cam_instance.get_frame.assert_called_once()
                mock_depth_processor.predict.assert_called_once_with(mock_frame)
                
                squeezed_raw_output = np.squeeze(mock_raw_depth_output)
                call_args_list = MockConvertToAbsoluteDepth.call_args_list
                
                assert len(call_args_list) > 0, "convert_to_absolute_depth was not called"
                args, kwargs = call_args_list[0]
                assert np.array_equal(args[0], squeezed_raw_output)
                assert args[1] == mock_config_data_no_compression 
                assert kwargs.get("is_compressed_grid") == False

                # Since grid_compression is explicitly disabled for this test run
                assert mock_config_data_no_compression.get("grid_compression", {}).get("enabled") is False
                # Check that processing_time_compression is present and is 0.0 or very small,
                # as the logic in linux_main.py calculates it even if compression is skipped.
                assert "processing_time_compression" in data
                assert data["processing_time_compression"] >= 0.0 # It can be a very small float
                
                mock_depth_processor.compress_depth_to_grid.assert_not_called()
                assert len(call_args_list) == 1, "convert_to_absolute_depth called more than once when compression is off"

                MockDepthToPointCloud.assert_called_once()
                args_pc, kwargs_pc = MockDepthToPointCloud.call_args
                
                assert not args_pc, "Expected no positional arguments for depth_to_point_cloud"
                assert np.array_equal(kwargs_pc.get('depth_data'), mock_depth_map_for_conversion)
                assert kwargs_pc.get('camera_intrinsics') == mock_config_data_no_compression["point_cloud"]["camera_intrinsics"]
                assert kwargs_pc.get('is_grid_data') is False # Explicitly False due to config
                expected_original_image_dims = (mock_config_data_no_compression["camera"]["height"], mock_config_data_no_compression["camera"]["width"])
                assert kwargs_pc.get('original_image_dims') == expected_original_image_dims
                assert kwargs_pc.get('grid_config') is None # Explicitly None due to config

