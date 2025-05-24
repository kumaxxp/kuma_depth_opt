\
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import time

# Add the project root to the path to allow importing windows_client
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from windows_client import DataReceiver, PCVisualizer, 안전한_json_파싱

# --- Fixtures ---

@pytest.fixture
def mock_windows_config():
    return {
        "server_address": "http://localhost:8000/pointcloud",
        "request_interval_ms": 100, # 10fps
        "visualization": {
            "enabled": True,
            "plot_title": "Real-time 3D Point Cloud",
            "point_size": 1,
            "x_min": -5.0, "x_max": 5.0,
            "y_min": -5.0, "y_max": 5.0,
            "z_min": 0.0, "z_max": 10.0,
            "view_angle": {"elev": 30, "azim": 45},
            "update_interval_ms": 50
        },
        "logging": {"level": "INFO"}
    }

@pytest.fixture
def data_receiver(mock_windows_config):
    with patch("windows_client.load_config", return_value=mock_windows_config):
        receiver = DataReceiver(config_path="dummy_config.json")
    return receiver

@pytest.fixture
def pc_visualizer(mock_windows_config):
    # PCVisualizer might try to create a plot, mock matplotlib
    with patch("windows_client.plt") as mock_plt, \
         patch("windows_client.load_config", return_value=mock_windows_config):
        visualizer = PCVisualizer(config_path="dummy_config.json")
        visualizer.plt = mock_plt # Attach the mock for assertions
    return visualizer

# --- Tests for 안전한_json_파싱 ---

def test_안전한_json_파싱_valid_json():
    json_string = '{"key": "value", "number": 123}'
    expected_data = {"key": "value", "number": 123}
    assert 안전한_json_파싱(json_string) == expected_data

def test_안전한_json_파싱_invalid_json():
    json_string = '{"key": "value", "number": 123' # Missing closing brace
    assert 안전한_json_파싱(json_string) is None

def test_안전한_json_파싱_empty_string():
    assert 안전한_json_파싱("") is None

def test_안전한_json_파싱_none_input():
    assert 안전한_json_파싱(None) is None

# --- Tests for DataReceiver ---

@patch("windows_client.requests.get")
def test_data_receiver_fetch_data_success(mock_get, data_receiver, mock_windows_config):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_point_cloud_data = {"point_cloud": [[1.0, 2.0, 3.0]], "timestamp_capture": time.time()}
    mock_response.text = '{"point_cloud": [[1.0, 2.0, 3.0]], "timestamp_capture": ' + str(mock_point_cloud_data["timestamp_capture"]) + '}'
    mock_get.return_value = mock_response

    data = data_receiver.fetch_data()
    
    mock_get.assert_called_once_with(mock_windows_config["server_address"], timeout=5)
    assert data is not None
    assert data["point_cloud"] == [[1.0, 2.0, 3.0]]
    assert "timestamp_capture" in data

@patch("windows_client.requests.get")
def test_data_receiver_fetch_data_request_exception(mock_get, data_receiver):
    mock_get.side_effect = Exception("Test network error")
    data = data_receiver.fetch_data()
    assert data is None

@patch("windows_client.requests.get")
def test_data_receiver_fetch_data_bad_status_code(mock_get, data_receiver):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response
    data = data_receiver.fetch_data()
    assert data is None

@patch("windows_client.requests.get")
def test_data_receiver_fetch_data_invalid_json(mock_get, data_receiver):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "invalid json"
    mock_get.return_value = mock_response
    data = data_receiver.fetch_data()
    assert data is None

# --- Tests for PCVisualizer ---

def test_pc_visualizer_initialization(pc_visualizer, mock_windows_config):
    assert pc_visualizer.config == mock_windows_config
    assert pc_visualizer.point_size == mock_windows_config["visualization"]["point_size"]
    pc_visualizer.plt.ion.assert_called_once()
    pc_visualizer.plt.figure.assert_called_once()
    
    # Check if add_subplot was called on the figure object
    fig_instance = pc_visualizer.plt.figure.return_value
    fig_instance.add_subplot.assert_called_once_with(111, projection='3d')

    ax_instance = fig_instance.add_subplot.return_value
    ax_instance.set_xlabel.assert_called_once_with('X (m)')
    ax_instance.set_ylabel.assert_called_once_with('Y (m)')
    ax_instance.set_zlabel.assert_called_once_with('Z (m)')
    ax_instance.set_title.assert_called_once_with(mock_windows_config["visualization"]["plot_title"])


def test_pc_visualizer_update_plot_no_data(pc_visualizer):
    pc_visualizer.update_plot(None)
    pc_visualizer.ax.clear.assert_called_once() # Should clear if no data
    # Check if axis limits and labels are reset
    vis_config = pc_visualizer.config["visualization"]
    pc_visualizer.ax.set_xlim.assert_called_once_with([vis_config["x_min"], vis_config["x_max"]])
    # ... (assert other set_xlim, set_ylim, set_zlim, set_xlabel, etc. calls for resetting the plot)
    pc_visualizer.plt.draw.assert_called_once()
    pc_visualizer.plt.pause.assert_called_with(0.001)


def test_pc_visualizer_update_plot_with_data(pc_visualizer, mock_windows_config):
    point_cloud_data = {
        "point_cloud": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "timestamp_capture": time.time(),
        "processing_time_total": 0.1
    }
    pc_visualizer.update_plot(point_cloud_data)

    pc_visualizer.ax.clear.assert_called_once()
    
    # Check scatter call
    points_array = np.array(point_cloud_data["point_cloud"])
    # Need to access the call_args of the mock
    args, kwargs = pc_visualizer.ax.scatter.call_args
    assert np.array_equal(args[0], points_array[:, 0]) # X
    assert np.array_equal(args[1], points_array[:, 1]) # Y
    assert np.array_equal(args[2], points_array[:, 2]) # Z
    assert kwargs.get('s') == mock_windows_config["visualization"]["point_size"]
    assert kwargs.get('c') == points_array[:, 2] # Color by Z
    assert kwargs.get('cmap') == "viridis"

    # Check title update
    expected_title_part = f"TS: {point_cloud_data['timestamp_capture']:.2f}, Proc: {point_cloud_data['processing_time_total']:.3f}s"
    
    # Get the title set by the last call to set_title
    # The first call is in __init__, the second in update_plot
    actual_title_call = pc_visualizer.ax.set_title.call_args_list[1]
    actual_title_args, _ = actual_title_call
    actual_title = actual_title_args[0]

    assert mock_windows_config["visualization"]["plot_title"] in actual_title
    assert expected_title_part in actual_title
    
    pc_visualizer.plt.draw.assert_called_once()
    pc_visualizer.plt.pause.assert_called_with(0.001)

def test_pc_visualizer_update_plot_empty_point_cloud(pc_visualizer):
    point_cloud_data = {"point_cloud": [], "timestamp_capture": time.time()}
    pc_visualizer.update_plot(point_cloud_data)
    pc_visualizer.ax.clear.assert_called_once()
    pc_visualizer.ax.scatter.assert_not_called() # Should not scatter if no points
    pc_visualizer.plt.draw.assert_called_once()

# --- Test main loop (conceptual, mocking time.sleep and external calls) ---

@patch("windows_client.DataReceiver")
@patch("windows_client.PCVisualizer")
@patch("windows_client.load_config")
@patch("windows_client.time.sleep") # Mock sleep
@patch("windows_client.logger") # Mock logger
def test_main_loop_logic(mock_logger, mock_sleep, mock_load_config, MockPCVisualizer, MockDataReceiver, mock_windows_config):
    # Setup mocks
    mock_load_config.return_value = mock_windows_config
    mock_receiver_instance = MockDataReceiver.return_value
    mock_visualizer_instance = MockPCVisualizer.return_value
    
    # Simulate a few iterations: 1 success, 1 no data
    mock_data_1 = {"point_cloud": [[1.0,2.0,3.0]], "timestamp_capture": 123.45}
    mock_data_2 = None # Simulate fetch failure
    
    # Configure side_effect to control loop termination after a few calls
    fetch_results = [mock_data_1, mock_data_2, KeyboardInterrupt("Stop test")] 
    mock_receiver_instance.fetch_data.side_effect = fetch_results

    # Run the main function from windows_client
    # Need to import it here to avoid issues with module-level mocks if main is run at import
    from windows_client import main 
    
    with pytest.raises(KeyboardInterrupt, match="Stop test"):
        main()

    # Assertions
    assert mock_receiver_instance.fetch_data.call_count == 3
    
    # Check calls to visualizer
    mock_visualizer_instance.update_plot.assert_any_call(mock_data_1)
    mock_visualizer_instance.update_plot.assert_any_call(mock_data_2) # update_plot should be called even with None
    assert mock_visualizer_instance.update_plot.call_count == 2 # Called for mock_data_1 and mock_data_2

    # Check sleep calls
    # Expected sleep time is config.request_interval_ms / 1000.0
    expected_sleep_duration = mock_windows_config["request_interval_ms"] / 1000.0
    # Sleep should have been called twice (after processing mock_data_1 and mock_data_2)
    mock_sleep.assert_has_calls([
        call(expected_sleep_duration),
        call(expected_sleep_duration)
    ])
    assert mock_sleep.call_count == 2

    # Check if visualization was disabled if config said so
    mock_windows_config_no_vis = mock_windows_config.copy()
    mock_windows_config_no_vis["visualization"]["enabled"] = False
    mock_load_config.return_value = mock_windows_config_no_vis
    
    # Reset mocks for a new run
    MockPCVisualizer.reset_mock()
    MockDataReceiver.reset_mock()
    mock_receiver_instance.fetch_data.side_effect = [KeyboardInterrupt("Stop test 2")] # Only one fetch

    with pytest.raises(KeyboardInterrupt, match="Stop test 2"):
        main()
    
    MockPCVisualizer.assert_not_called() # Visualizer should not be initialized
    mock_visualizer_instance.update_plot.assert_not_called() # update_plot should not be called

