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
    with patch("linux_main.config", None), \\
         patch("linux_main.depth_processor_instance", None), \\
         patch("linux_main.camera_capture", None):
        
        response = client.get("/pointcloud")
        assert response.status_code == 200 # Endpoint itself is up
        data = response.json()
        assert data["error_message"] == "System not initialized"
        assert data["point_cloud"] == []
        assert data["processing_time_total"] == 0

def test_get_pointcloud_success(client, mock_config_data):
    mock_frame = np.random.randint(0, 256, (mock_config_data["camera"]["height"], mock_config_data["camera"]["width"], 3), dtype=np.uint8)
    mock_depth_map = np.random.rand(mock_config_data["depth_model_parameters"]["input_height"], mock_config_data["depth_model_parameters"]["input_width"]).astype(np.float32)
    mock_point_cloud = [[float(i), float(i), float(i)] for i in range(5)] # Simple mock

    with patch("linux_main.load_config", return_value=mock_config_data), \\
         patch("linux_main.CamInput") as MockCamInput, \\
         patch("linux_main.initialize_depth_model") as MockInitializeDepthModel, \\
         patch("linux_main.depth_to_point_cloud", return_value=np.array(mock_point_cloud)) as MockDepthToPointCloud:

        # Configure CamInput mock
        mock_cam_instance = MockCamInput.return_value
        mock_cam_instance.get_frame.return_value = mock_frame
        mock_cam_instance.width = mock_config_data["camera"]["width"]
        mock_cam_instance.height = mock_config_data["camera"]["height"]

        # Configure DepthProcessor mock (returned by initialize_depth_model)
        mock_depth_processor = MockInitializeDepthModel.return_value
        mock_depth_processor.predict.return_value = mock_depth_map
        mock_depth_processor.compress_depth_to_grid.return_value = mock_depth_map # Assuming no compression for simplicity or mock it properly if needed

        # Simulate app startup to initialize globals based on mocks
        # This is a bit of a workaround. Ideally, TestClient would handle this,
        # but our startup logic is complex.
        with patch.dict(sys.modules):
            if 'linux_main' in sys.modules:
                del sys.modules['linux_main']
            
            # Re-import after deleting to re-trigger startup logic with mocks
            # This is still problematic as startup_event is FastAPI specific.
            # A better approach would be to refactor linux_main.py to make
            # initialization more testable (e.g., pass dependencies into functions).

            # For now, let's try to manually set the globals that startup_event would set.
            # This assumes that load_config is called within startup_event or before.
            
            # Manually trigger the parts of startup_event relevant for this test
            # This is still not ideal as we are replicating startup logic.
            
            # Instead of re-importing, let's try to patch the globals directly
            # after the client has been created, assuming TestClient(app) runs startup.
            # The challenge is that startup_event in linux_main.py uses the *actual*
            # functions, not the mocked ones, unless we can patch them *before*
            # TestClient(app) is called.

            # Let's assume the client fixture handles startup and our patches are applied.
            # We will patch the globals that are set by startup_event.
            
            # Patching globals directly within linux_main for the scope of this test
            with patch("linux_main.config", mock_config_data), \\
                 patch("linux_main.camera_capture", mock_cam_instance), \\
                 patch("linux_main.depth_processor_instance", mock_depth_processor):

                response = client.get("/pointcloud")
                assert response.status_code == 200
                data = response.json()

                assert data["error_message"] is None
                assert data["point_cloud"] == mock_point_cloud
                assert "processing_time_total" in data
                assert data["processing_time_total"] > 0
                assert "processing_time_capture_s" in data
                assert "processing_time_depth_s" in data
                assert "processing_time_pc_s" in data
                
                mock_cam_instance.get_frame.assert_called_once()
                mock_depth_processor.predict.assert_called_once_with(mock_frame)
                
                # Check if compress_depth_to_grid was called based on config
                if mock_config_data.get("grid_compression", {}).get("enabled", False):
                    mock_depth_processor.compress_depth_to_grid.assert_called_once_with(mock_depth_map)
                else:
                    mock_depth_processor.compress_depth_to_grid.assert_not_called()

                MockDepthToPointCloud.assert_called_once()
                # The actual arguments to depth_to_point_cloud can be complex to assert fully
                # due to potential transformations. Let's check the first arg (depth_map).
                args, kwargs = MockDepthToPointCloud.call_args
                assert np.array_equal(args[0], mock_depth_map)
                assert args[1] == mock_config_data["point_cloud"]["camera_intrinsics"]
                assert args[2] == (mock_config_data["camera"]["height"], mock_config_data["camera"]["width"])


# More tests will be added here.
# For example, a test for a successful point cloud generation:
# def test_get_pointcloud_success(client, mock_config_data):
#     # This test requires mocking:
#     # - load_config to return mock_config_data
#     # - CamInput and its get_frame method
#     # - initialize_depth_model and its predict/compress_depth_to_grid methods
#     # - depth_to_point_cloud
#     pass

