\
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import time
import requests.exceptions # Added for RequestException

# Add the project root to the path to allow importing windows_client
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from windows_client import DataReceiver, PCVisualizer, safe_json_parse

# --- Fixtures ---

@pytest.fixture
def mock_windows_config(): # Updated to match actual config_windows.json structure and windows_client.py usage
    return {
        "server_connection": {
            "linux_server_ip": "localhost",
            "linux_server_port": 8000
        },
        "client": {
            "request_timeout_s": 3.0,
            "polling_interval_ms": 150 # For FuncAnimation interval
        },
        "visualization": {
            # "enabled": True, # Not directly used by PCVisualizer or DataReceiver constructors
            # "plot_title": "Real-time 3D Point Cloud", # Title is hardcoded in PCVisualizer
            "point_size": 2, # Changed for clarity in test
            "plot_limit_x_m": [-3, 3], # Changed for clarity
            "plot_limit_y_m": [-3, 3], # Changed for clarity
            "plot_limit_z_m": [0, 6],  # Changed for clarity
            "view_elevation": 25,      # Changed for clarity
            "view_azimuth": -50,     # Changed for clarity
            "figure_size": [12, 9]   # Changed for clarity
        },
        "logging": {"level": "DEBUG"} # Changed for clarity
    }

@pytest.fixture
def data_receiver(mock_windows_config): # Corrected initialization
    server_conf = mock_windows_config.get("server_connection", {})
    client_conf = mock_windows_config.get("client", {})
    
    server_ip = server_conf.get("linux_server_ip", "localhost")
    server_port = server_conf.get("linux_server_port", 8000)
    base_url = f"http://{server_ip}:{server_port}"
    # Use timeout from client_conf, or default if not present
    timeout = client_conf.get("request_timeout_s", 5.0) 
    
    # server_url, endpoint, request_timeout_s
    receiver = DataReceiver(server_url=base_url, endpoint="/pointcloud", request_timeout_s=timeout)
    return receiver

@pytest.fixture
def pc_visualizer(mock_windows_config): # Corrected initialization
    # PCVisualizer needs a fig, ax, and its specific part of the config
    with patch("windows_client.plt.figure") as mock_plt_figure_call:
        mock_fig_instance = mock_plt_figure_call.return_value
        mock_ax_instance = mock_fig_instance.add_subplot.return_value
        
        # Each call to fig.text() in PCVisualizer.__init__ needs to return a *distinct* mock text object.
        # These text objects will then have their own .set_text() mock methods.
        mock_fps_text_obj = MagicMock(name="fps_text_obj")
        mock_timestamp_text_obj = MagicMock(name="timestamp_text_obj")
        mock_points_text_obj = MagicMock(name="points_text_obj")
        mock_connection_status_text_obj = MagicMock(name="connection_status_text_obj")

        # Configure the mock_fig_instance.text method to return these mocks 
        # in the order they are called in PCVisualizer.__init__:
        # 1. self.fps_text
        # 2. self.timestamp_text
        # 3. self.points_text
        # 4. self.connection_status_text
        mock_fig_instance.text.side_effect = [
            mock_fps_text_obj,
            mock_timestamp_text_obj,
            mock_points_text_obj,
            mock_connection_status_text_obj
        ]
        
        vis_conf = mock_windows_config.get("visualization", {})
        visualizer = PCVisualizer(fig=mock_fig_instance, ax=mock_ax_instance, vis_config=vis_conf)
        
        # For clarity in tests, you could also re-assign these if needed, though direct access is fine:
        # visualizer.fps_text = mock_fps_text_obj
        # visualizer.timestamp_text = mock_timestamp_text_obj
        # visualizer.points_text = mock_points_text_obj
        # visualizer.connection_status_text = mock_connection_status_text_obj
        
        # Store mocks for assertion if needed, though visualizer.ax etc. can be used
        visualizer._test_mock_fig = mock_fig_instance
        visualizer._test_mock_ax = mock_ax_instance
    return visualizer

# --- Tests for safe_json_parse ---

def test_safe_json_parse_valid_json():
    json_string = '{"key": "value", "number": 123}'
    expected_data = {"key": "value", "number": 123}
    assert safe_json_parse(json_string) == expected_data

def test_safe_json_parse_invalid_json():
    json_string = '{"key": "value", "number": 123' # Missing closing brace
    assert safe_json_parse(json_string) is None

def test_safe_json_parse_empty_string():
    assert safe_json_parse("") is None

def test_safe_json_parse_none_input():
    assert safe_json_parse(None) is None

# --- Tests for DataReceiver ---

@patch("windows_client.requests.get")
def test_data_receiver_fetch_data_success(mock_get, data_receiver): # Removed mock_windows_config as it's not directly used
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Use a more realistic timestamp and ensure it's a float for JSON serialization
    timestamp_val = time.time()
    mock_point_cloud_data = {"point_cloud": [[1.0, 2.0, 3.0]], "timestamp_capture": timestamp_val}
    # Correctly format the JSON string
    mock_response.text = f'{{"point_cloud": [[1.0, 2.0, 3.0]], "timestamp_capture": {timestamp_val}}}'
    mock_get.return_value = mock_response

    data = data_receiver.fetch_data()
    
    # Assert against data_receiver's configured properties (url, request_timeout_s)
    mock_get.assert_called_once_with(data_receiver.url, timeout=data_receiver.request_timeout_s)
    assert data is not None
    assert data["point_cloud"] == [[1.0, 2.0, 3.0]]
    assert data["timestamp_capture"] == timestamp_val

@patch("windows_client.requests.get")
def test_data_receiver_fetch_data_request_exception(mock_get, data_receiver):
    mock_get.side_effect = requests.exceptions.RequestException("Test network error") # Changed to specific exception
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
    # pc_visualizer fixture now creates a valid instance with mocked fig and ax
    vis_config_from_fixture = mock_windows_config["visualization"]
    
    assert pc_visualizer.point_size == vis_config_from_fixture["point_size"]
    assert pc_visualizer.plot_limits["x"] == vis_config_from_fixture["plot_limit_x_m"]
    assert pc_visualizer.plot_limits["y"] == vis_config_from_fixture["plot_limit_y_m"]
    assert pc_visualizer.plot_limits["z"] == vis_config_from_fixture["plot_limit_z_m"]

    # Get the ax mock from the visualizer instance itself
    ax_mock = pc_visualizer.ax

    ax_mock.set_xlabel.assert_called_once_with("X (m)")
    ax_mock.set_ylabel.assert_called_once_with("Y (m)")
    ax_mock.set_zlabel.assert_called_once_with("Z (m)")
    ax_mock.set_title.assert_called_once_with("Real-time 3D Point Cloud") # Title is hardcoded in PCVisualizer

    ax_mock.set_xlim.assert_called_once_with(vis_config_from_fixture["plot_limit_x_m"])
    ax_mock.set_ylim.assert_called_once_with(vis_config_from_fixture["plot_limit_y_m"])
    ax_mock.set_zlim.assert_called_once_with(vis_config_from_fixture["plot_limit_z_m"])
    ax_mock.view_init.assert_called_once_with(elev=vis_config_from_fixture["view_elevation"], azim=vis_config_from_fixture["view_azimuth"])

    # plt.ion() is not called by PCVisualizer constructor
    # Figure and add_subplot are called during fixture setup, not directly by PCVisualizer constructor test here.
    # Those calls are implicitly tested by the fixture setup itself.

def test_pc_visualizer_update_plot_no_data(pc_visualizer):
    # Mock the data_receiver that update_plot will use
    mock_data_receiver = MagicMock(spec=DataReceiver)
    mock_data_receiver.fetch_data.return_value = None # Simulate no data

    pc_visualizer.update_plot(frame=0, data_receiver=mock_data_receiver) # Pass the mocked receiver

    # Check that fetch_data was called
    mock_data_receiver.fetch_data.assert_called_once()

    # Assert calls to set_text
    pc_visualizer.points_text.set_text.assert_called_with("Points: 0 (no data)")
    pc_visualizer.connection_status_text.set_text.assert_called_with("Status: Disconnected/No Data")
    pc_visualizer.timestamp_text.set_text.assert_called_with("Server Timestamp: N/A")

def test_pc_visualizer_update_plot_with_data(pc_visualizer, mock_windows_config):
    mock_data_receiver = MagicMock(spec=DataReceiver)
    point_cloud_raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    server_timestamp = "2023-10-27T10:00:00Z"
    mock_server_data = {
        "point_cloud": point_cloud_raw,
        "timestamp_processing_end": server_timestamp 
        # Add other fields if your actual data has them and they are used
    }
    mock_data_receiver.fetch_data.return_value = mock_server_data

    # Mock the scatter method on the visualizer's ax object
    # pc_visualizer.ax is already a mock from the fixture setup
    pc_visualizer.ax.scatter = MagicMock(return_value=MagicMock()) # scatter returns an artist

    # If a previous scatter existed, it should be removed.
    # Let's assume self.scatter is initially None or an old mock that needs .remove()
    if pc_visualizer.scatter: # If it's not None
        pc_visualizer.scatter.remove = MagicMock()


    pc_visualizer.update_plot(frame=0, data_receiver=mock_data_receiver)

    mock_data_receiver.fetch_data.assert_called_once()
    
    # Check scatter call (points are filtered by plot limits in update_plot)
    # For this test, assume points are within limits defined in mock_windows_config["visualization"]
    # plot_limit_x_m: [-3, 3], plot_limit_y_m: [-3, 3], plot_limit_z_m: [0, 6]
    # Points [1,2,3] and [4,5,6] are within these if x,y are within [-3,3] and z within [0,6]
    # Point [4,5,6] y=5 is outside [-3,3] if limits are strict. Let's adjust points or limits for test.
    # Using points: [[1,1,1], [0,0,2]] which are within default test limits.
    point_cloud_raw_valid = [[1.0,1.0,1.0], [0.0,0.0,2.0]]
    mock_server_data_valid = { "point_cloud": point_cloud_raw_valid, "timestamp_processing_end": server_timestamp }
    mock_data_receiver.fetch_data.return_value = mock_server_data_valid
    
    # Reset scatter mock and call again with valid data
    pc_visualizer.ax.scatter = MagicMock(return_value=MagicMock())
    if pc_visualizer.scatter and hasattr(pc_visualizer.scatter, 'remove'):
        pc_visualizer.scatter.remove = MagicMock() # mock remove if called
    else: # If scatter was None or not a mock with remove
        pass


    pc_visualizer.update_plot(frame=0, data_receiver=mock_data_receiver) # Call again

    points_array = np.array(point_cloud_raw_valid)
    
    # Check that ax.scatter was called
    pc_visualizer.ax.scatter.assert_called_once()
    args, kwargs = pc_visualizer.ax.scatter.call_args
    
    assert np.array_equal(args[0], points_array[:, 0]) # X
    assert np.array_equal(args[1], points_array[:, 1]) # Y
    assert np.array_equal(args[2], points_array[:, 2]) # Z
    assert kwargs.get('s') == mock_windows_config["visualization"]["point_size"]
    assert np.array_equal(kwargs.get('c'), points_array[:, 2]) # Color by Z
    assert kwargs.get('cmap') == "viridis"

    # Assert calls to set_text
    pc_visualizer.points_text.set_text.assert_called_with(f"Points: {len(points_array)}")
    pc_visualizer.connection_status_text.set_text.assert_called_with("Status: Connected")
    pc_visualizer.timestamp_text.set_text.assert_called_with(f"Server Timestamp: {server_timestamp}")

def test_pc_visualizer_update_plot_empty_point_cloud(pc_visualizer):
    mock_data_receiver = MagicMock(spec=DataReceiver)
    mock_data_receiver.fetch_data.return_value = {"point_cloud": [], "timestamp_processing_end": "N/A"}

    pc_visualizer.ax.scatter = MagicMock() # Ensure scatter is a mock to check not_called

    pc_visualizer.update_plot(frame=0, data_receiver=mock_data_receiver)
    
    pc_visualizer.ax.scatter.assert_not_called()
    # Assert calls to set_text
    # Corrected expected text to match actual behavior for an empty point cloud list
    pc_visualizer.points_text.set_text.assert_called_with("Points: 0 (no data)")

# --- Test main loop ---

@patch("windows_client.load_config")      # Outermost patch, will be last mock param before fixture
@patch("windows_client.DataReceiver")      # Next, and so on
@patch("windows_client.PCVisualizer")
@patch("windows_client.plt.figure")
@patch("windows_client.plt.show")
@patch("windows_client.FuncAnimation")
@patch("windows_client.logger")           # Innermost patch, will be first mock param
def test_main_loop_logic(
    mock_logger_param,          # Corresponds to @patch("windows_client.logger")
    MockFuncAnimation_param,    # Corresponds to @patch("windows_client.FuncAnimation")
    mock_plt_show_param,        # Corresponds to @patch("windows_client.plt.show")
    mock_plt_figure_param,      # Corresponds to @patch("windows_client.plt.figure")
    MockPCVisualizer_param,     # Corresponds to @patch("windows_client.PCVisualizer")
    MockDataReceiver_param,     # Corresponds to @patch("windows_client.DataReceiver")
    patched_load_config_param,  # Corresponds to @patch("windows_client.load_config")
    mock_windows_config         # Fixture (name must match fixture defined above)
):
    # Setup mocks
    patched_load_config_param.return_value = mock_windows_config # Use fixture value

    # .return_value gives the instance created when the patched class is called
    mock_receiver_instance = MockDataReceiver_param.return_value
    mock_visualizer_instance = MockPCVisualizer_param.return_value
    mock_fig_instance = mock_plt_figure_param.return_value
    # If add_subplot is called on fig_instance, it also needs to be a mock or configured
    mock_ax_instance = mock_fig_instance.add_subplot.return_value

    # Make plt.show() raise KeyboardInterrupt to terminate main()
    mock_plt_show_param.side_effect = KeyboardInterrupt("Stop test from plt.show")

    # Import main locally to ensure it uses the patched versions
    from windows_client import main 
    
    with pytest.raises(KeyboardInterrupt, match="Stop test from plt.show"):
        main()

    # Assertions
    patched_load_config_param.assert_called_once_with("config_windows.json")

    # Check DataReceiver instantiation
    server_cfg = mock_windows_config.get("server_connection", {})
    client_cfg = mock_windows_config.get("client", {})
    expected_server_url = f"http://{server_cfg.get('linux_server_ip', 'localhost')}:{server_cfg.get('linux_server_port', 8000)}"
    expected_timeout = client_cfg.get('request_timeout_s', 5.0) # Default from DataReceiver if not in config
    MockDataReceiver_param.assert_called_once_with(
        server_url=expected_server_url, 
        endpoint="/pointcloud", 
        request_timeout_s=expected_timeout
    )

    # Check figure and PCVisualizer instantiation
    vis_cfg = mock_windows_config.get("visualization", {})
    expected_figure_size = vis_cfg.get("figure_size", (10,8)) # Default from windows_client.py main()
    mock_plt_figure_param.assert_called_once_with(figsize=expected_figure_size)
    mock_fig_instance.add_subplot.assert_called_once_with(111, projection='3d')
    # Corrected: Call with positional arguments as in main()
    MockPCVisualizer_param.assert_called_once_with(mock_fig_instance, mock_ax_instance, vis_cfg)

    # Check FuncAnimation instantiation
    expected_polling_interval = client_cfg.get("polling_interval_ms", 200) # Default from windows_client.py main()
    MockFuncAnimation_param.assert_called_once()
    # FuncAnimation(fig, visualizer.update_plot, fargs=(data_receiver,), interval=polling_interval, blit=False, cache_frame_data=False)
    args_fa, kwargs_fa = MockFuncAnimation_param.call_args
    assert args_fa[0] == mock_fig_instance                            # fig
    assert args_fa[1] == mock_visualizer_instance.update_plot         # func
    assert kwargs_fa['fargs'] == (mock_receiver_instance,)            # fargs
    assert kwargs_fa['interval'] == expected_polling_interval         # interval
    assert kwargs_fa['blit'] is False                                 # blit
    assert kwargs_fa['cache_frame_data'] is False                     # cache_frame_data
    
    mock_plt_show_param.assert_called_once()
    # mock_logger_param is available if you need to assert logging calls.

