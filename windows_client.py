import matplotlib.pyplot as plt
import numpy as np
import requests
import time
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import logging

# Import utility for loading configuration
from utils import load_config

# Logger setup
logger = logging.getLogger("kuma_depth_opt.windows_client")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DataReceiver:
    """Handles fetching point cloud data from the Linux server."""
    def __init__(self, server_url: str, endpoint: str = "/pointcloud", request_timeout_s: float = 5.0): # Added request_timeout_s
        self.base_url = server_url.rstrip('/')
        self.endpoint = endpoint
        self.url = f"{self.base_url}{self.endpoint}"
        self.request_timeout_s = request_timeout_s # Store timeout
        logger.info(f"DataReceiver initialized. Target URL: {self.url}, Timeout: {self.request_timeout_s}s")

    def fetch_data(self) -> dict | None:
        """Fetches point cloud data from the server."""
        try:
            response = requests.get(self.url, timeout=self.request_timeout_s) # Use stored timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            logger.debug(f"Successfully fetched data: {len(data.get('point_cloud', []))} points, timestamp: {data.get('timestamp_processing_end')}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {self.url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from {self.url}: {e}")
            return None

class PCVisualizer:
    """Visualizes the point cloud data in a real-time 3D plot."""
    def __init__(self, fig, ax, vis_config: dict): # Changed config to vis_config for clarity
        self.fig = fig
        self.ax = ax
        self.scatter = None
        # Use direct keys from vis_config as per new config_windows.json structure
        self.plot_limits = {
            "x": vis_config.get("plot_limit_x_m", [-2, 2]),
            "y": vis_config.get("plot_limit_y_m", [-2, 2]),
            "z": vis_config.get("plot_limit_z_m", [0, 5])
        }
        self.point_size = vis_config.get("point_size", 1)
        # animation_interval will be handled in main based on client_config

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("Real-time 3D Point Cloud")
        self.ax.set_xlim(self.plot_limits["x"])
        self.ax.set_ylim(self.plot_limits["y"])
        self.ax.set_zlim(self.plot_limits["z"])
        # Invert Y-axis to match camera view (optional, depends on coordinate system)
        # self.ax.invert_yaxis()
        # Invert Z-axis if depth is positive away from camera but you want Z up (also optional)
        # self.ax.invert_zaxis()
        self.ax.view_init(elev=vis_config.get("view_elevation", 20), azim=vis_config.get("view_azimuth", -60))

        # For FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_text = self.fig.text(0.02, 0.95, "", transform=self.fig.transFigure)
        self.timestamp_text = self.fig.text(0.02, 0.90, "", transform=self.fig.transFigure)
        self.points_text = self.fig.text(0.02, 0.85, "", transform=self.fig.transFigure)
        self.connection_status_text = self.fig.text(0.02, 0.80, "Status: Connecting...", transform=self.fig.transFigure)

        logger.info("PCVisualizer initialized.")

    def update_plot(self, frame, data_receiver: DataReceiver): # MODIFIED SIGNATURE
        """Updates the 3D scatter plot with new point cloud data."""
        # frame is the frame number/data from FuncAnimation, can be ignored if not used
        # data_receiver is the DataReceiver instance passed via fargs
        
        data = data_receiver.fetch_data()

        if data and 'point_cloud' in data and data['point_cloud']:
            points = np.array(data['point_cloud'])
            if points.ndim == 2 and points.shape[1] == 3:
                if self.scatter is not None:
                    self.scatter.remove()
                
                # Filter points based on plot_limits for cleaner visualization
                x_lim, y_lim, z_lim = self.plot_limits['x'], self.plot_limits['y'], self.plot_limits['z']
                mask = (points[:,0] >= x_lim[0]) & (points[:,0] <= x_lim[1]) & \
                       (points[:,1] >= y_lim[0]) & (points[:,1] <= y_lim[1]) & \
                       (points[:,2] >= z_lim[0]) & (points[:,2] <= z_lim[1])
                filtered_points = points[mask]

                if filtered_points.size > 0:
                    self.scatter = self.ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], s=self.point_size, c=filtered_points[:, 2], cmap='viridis')
                    self.points_text.set_text(f"Points: {len(filtered_points)}")
                else:
                    # Clear previous points if no valid points are received
                    if self.scatter is not None:
                        self.scatter.remove()
                        self.scatter = None # Ensure it's None so a new one is created next time
                    self.points_text.set_text("Points: 0 (all filtered or empty)")
                
                self.connection_status_text.set_text("Status: Connected")
                self.connection_status_text.set_color("green")
                
                # Update timestamp from server
                proc_end_ts = data.get('timestamp_processing_end', 'N/A')
                self.timestamp_text.set_text(f"Server Timestamp: {proc_end_ts}")

            else:
                logger.warning(f"Received point cloud data is not in the expected format (N, 3). Shape: {points.shape}")
                self.points_text.set_text("Points: 0 (invalid format)")
                self.connection_status_text.set_text("Status: Data format error")
                self.connection_status_text.set_color("red")
        else:
            logger.warning("No point cloud data received or data is empty.")
            # Clear previous points if no data is received
            if self.scatter is not None:
                self.scatter.remove()
                self.scatter = None
            self.points_text.set_text("Points: 0 (no data)")
            self.connection_status_text.set_text("Status: Disconnected/No Data")
            self.connection_status_text.set_color("orange")
            self.timestamp_text.set_text("Server Timestamp: N/A")

        # Calculate and display FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            self.fps_text.set_text(f"Client FPS: {fps:.2f}")
        
        return self.scatter, self.fps_text, self.timestamp_text, self.points_text, self.connection_status_text

def main():
    config_path = "config_windows.json"
    config = load_config(config_path)
    if not config:
        logger.error(f"Failed to load configuration from {config_path}. Exiting.")
        return

    server_config = config.get("server_connection", {}) # Still using server_connection for base URL
    client_config = config.get("client", {})
    vis_config = config.get("visualization", {})

    server_ip = server_config.get("linux_server_ip", "localhost")
    server_port = server_config.get("linux_server_port", 8000)
    server_url = f"http://{server_ip}:{server_port}"
    
    request_timeout = client_config.get("request_timeout_s", 5.0)
    polling_interval = client_config.get("polling_interval_ms", 200)

    data_receiver = DataReceiver(server_url=server_url, endpoint="/pointcloud", request_timeout_s=request_timeout)
    
    fig = plt.figure(figsize=vis_config.get("figure_size", (10, 8))) # figure_size might not be in new config, added default
    ax = fig.add_subplot(111, projection='3d')
    
    visualizer = PCVisualizer(fig, ax, vis_config) # Pass only visualization part of config

    ani = FuncAnimation(fig, visualizer.update_plot, fargs=(data_receiver,), 
                        interval=polling_interval, blit=False, cache_frame_data=False)
    
    plt.show()
    logger.info("Windows client application stopped.")

if __name__ == "__main__":
    main()
