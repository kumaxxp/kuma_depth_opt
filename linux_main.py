\
# filepath: c:\\work\\kuma_depth_opt\\linux_main.py
import asyncio
import cv2
import numpy as np
import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Optional

try:
    from utils import load_config
except ImportError:
    # This is to help locate utils.py if linux_main.py is run directly from its folder
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import load_config

from depth_processor.depth_model import initialize_depth_model, convert_to_absolute_depth
from depth_processor.point_cloud import depth_to_point_cloud

# --- Logger Setup ---
logger = logging.getLogger("linux_main")
logger.setLevel(logging.INFO) # Default to INFO, can be changed by config if needed
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# --- Configuration ---
CONFIG_PATH = "config_linux.json" 
config = None
depth_processor_instance = None
camera_capture = None

# --- Pydantic Models for API ---
class PointCloudResponse(BaseModel):
    timestamp_capture: float
    timestamp_processed: float
    processing_time_depth: float
    processing_time_compression: float
    processing_time_pointcloud: float
    processing_time_total: float
    point_cloud: List[Tuple[float, float, float]]
    error_message: Optional[str] = None

# --- FastAPI App ---
app = FastAPI(title="Depth Point Cloud API")

# --- Modules ---
class CamInput:
    def __init__(self, camera_config: dict):
        self.config = camera_config
        self.device_id = self.config.get("device_id", 0)
        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self.fps = self.config.get("fps", 10) # Note: FPS setting might not be strictly enforced

        logger.info(f"Attempting to initialize camera ID: {self.device_id} with {self.width}x{self.height} @ {self.fps} FPS")
        self.cap = cv2.VideoCapture(self.device_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera device ID: {self.device_id}")
            raise IOError(f"Cannot open camera {self.device_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera initialized. Requested: {self.width}x{self.height} @ {self.fps}FPS. Actual: {actual_width}x{actual_height} @ {actual_fps}FPS.")

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to grab frame from camera.")
            return None
        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    global config, depth_processor_instance, camera_capture
    try:
        logger.info("Linux module starting up...")
        config = load_config(CONFIG_PATH)
        if not config:
            logger.critical(f"CRITICAL: Failed to load configuration from {CONFIG_PATH}. Application cannot start correctly.")
            raise RuntimeError(f"Configuration load failed from {CONFIG_PATH}")
        logger.info("Configuration loaded successfully.")
        
        # Set log level from config if specified
        log_level_str = config.get("logging", {}).get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
        for handler in logger.handlers: # Apply to existing handlers too
            handler.setLevel(log_level)
        logger.info(f"Logger level set to {log_level_str}")

        # depth_model_cfg = config.get("depth_model", {}) # Not used directly here
        depth_processor_instance = initialize_depth_model(config) # Pass the whole config
        
        if not depth_processor_instance or not depth_processor_instance.is_available():
            logger.warning("Depth processor could not be initialized or is not available. API might return dummy data or limited functionality.")
        else:
            logger.info("Depth processor initialized successfully.")

        camera_cfg = config.get("camera")
        if camera_cfg:
            try:
                camera_capture = CamInput(camera_cfg)
                logger.info("Camera input module initialized.")
            except IOError as e:
                logger.error(f"Failed to initialize camera: {e}. Camera input will not be available.")
                camera_capture = None # Ensure it's None if failed
        else:
            logger.error("Camera configuration not found. Camera input will not be available.")
            camera_capture = None

    except Exception as e:
        logger.critical(f"Critical error during startup: {e}", exc_info=True)
        # Depending on deployment, might want to sys.exit(1) or let orchestrator handle
        raise # Re-raise to stop FastAPI startup if critical

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Linux module shutting down...")
    if camera_capture:
        camera_capture.release()
    logger.info("Shutdown complete.")

# --- FastAPI Endpoint ---
@app.get("/pointcloud", response_model=PointCloudResponse)
async def get_pointcloud_data():
    global config, depth_processor_instance, camera_capture # Read globals
    
    # Basic check for initialization
    if not all([config, depth_processor_instance, camera_capture]):
        logger.error("System not initialized properly (config, depth processor, or camera missing). Cannot process request.")
        return PointCloudResponse(
            timestamp_capture=time.time(), timestamp_processed=time.time(),
            processing_time_depth=0, processing_time_compression=0,
            processing_time_pointcloud=0, processing_time_total=0,
            point_cloud=[], error_message="System not initialized"
        )

    t_start_total = time.perf_counter()
    timestamp_capture = time.time() # Record wall-clock time at start of capture attempt

    # 1. Capture Frame
    frame = camera_capture.get_frame()
    if frame is None:
        logger.warning("Failed to get frame for point cloud generation.")
        return PointCloudResponse(
            timestamp_capture=timestamp_capture, timestamp_processed=time.time(),
            processing_time_depth=0, processing_time_compression=0,
            processing_time_pointcloud=0, processing_time_total=time.perf_counter() - t_start_total,
            point_cloud=[], error_message="Failed to capture frame"
        )
    
    # 2. Depth Estimation
    t_depth_start = time.perf_counter()
    relative_depth_map, _ = depth_processor_instance.predict(frame)
    if relative_depth_map is None:
        logger.warning("Depth estimation failed (returned None).")
        return PointCloudResponse(
            timestamp_capture=timestamp_capture, timestamp_processed=time.time(),
            processing_time_depth=time.perf_counter() - t_depth_start, 
            processing_time_compression=0, processing_time_pointcloud=0, 
            processing_time_total=time.perf_counter() - t_start_total,
            point_cloud=[], error_message="Depth estimation failed"
        )
    t_depth_end = time.perf_counter()
    
    # 3. Grid Compression (if enabled) & Absolute Depth Conversion
    t_compress_start = time.perf_counter()
    grid_compression_config = config.get("grid_compression", {})
    point_cloud_config = config.get("point_cloud", {})
    camera_intrinsics = point_cloud_config.get("camera_intrinsics")
    
    depth_for_pointcloud_conversion: Optional[np.ndarray] = None
    is_input_grid_data_for_pc = False
    
    if grid_compression_config.get("enabled", False):
        # Compress the relative depth map first
        compressed_relative_grid = depth_processor_instance.compress_depth_to_grid(relative_depth_map)
        if compressed_relative_grid is None:
            logger.warning("Grid compression failed.")
            return PointCloudResponse(
                timestamp_capture=timestamp_capture, timestamp_processed=time.time(),
                processing_time_depth=t_depth_end - t_depth_start, 
                processing_time_compression=time.perf_counter() - t_compress_start, 
                processing_time_pointcloud=0, 
                processing_time_total=time.perf_counter() - t_start_total,
                point_cloud=[], error_message="Grid compression failed"
            )
        # Then convert the compressed relative grid to absolute depth
        depth_for_pointcloud_conversion = convert_to_absolute_depth(compressed_relative_grid, config, is_compressed_grid=True)
        is_input_grid_data_for_pc = True
    else:
        # Convert the full relative depth map to absolute depth
        absolute_depth_map_full = convert_to_absolute_depth(relative_depth_map, config, is_compressed_grid=False)
        # Ensure it's 2D (H,W) for point cloud function if not grid
        if absolute_depth_map_full is not None:
            if len(absolute_depth_map_full.shape) == 4 and absolute_depth_map_full.shape[0] == 1 and absolute_depth_map_full.shape[3] == 1:
                depth_for_pointcloud_conversion = absolute_depth_map_full.squeeze(axis=(0,3))
            elif len(absolute_depth_map_full.shape) == 3 and absolute_depth_map_full.shape[2] == 1:
                depth_for_pointcloud_conversion = absolute_depth_map_full.squeeze(axis=2)
            elif len(absolute_depth_map_full.shape) == 2:
                depth_for_pointcloud_conversion = absolute_depth_map_full
            else:
                logger.warning(f"Unexpected shape of full absolute depth map: {absolute_depth_map_full.shape}. Cannot proceed.")
                depth_for_pointcloud_conversion = None
        is_input_grid_data_for_pc = False

    if depth_for_pointcloud_conversion is None:
        logger.warning("Depth data for point cloud is None after compression/conversion step.")
        return PointCloudResponse(
            timestamp_capture=timestamp_capture, timestamp_processed=time.time(),
            processing_time_depth=t_depth_end - t_depth_start, 
            processing_time_compression=time.perf_counter() - t_compress_start,
            processing_time_pointcloud=0, 
            processing_time_total=time.perf_counter() - t_start_total,
            point_cloud=[], error_message="Depth processing for PC failed"
        )
    t_compress_end = time.perf_counter() # Includes absolute conversion time

    # 4. Convert to Point Cloud
    t_pc_start = time.perf_counter()
    
    grid_params_for_pc_func = grid_compression_config if is_input_grid_data_for_pc else None
    
    cam_dims_config = config.get("camera", {})
    original_image_width = cam_dims_config.get("width")
    original_image_height = cam_dims_config.get("height")
    original_image_dims_tuple = None
    if original_image_width and original_image_height:
        original_image_dims_tuple = (original_image_width, original_image_height)

    if is_input_grid_data_for_pc and not original_image_dims_tuple :
        logger.error("Original image dimensions not found in camera config, required for grid to point cloud projection.")
        return PointCloudResponse(
            timestamp_capture=timestamp_capture, timestamp_processed=time.time(),
            processing_time_depth=t_depth_end - t_depth_start, 
            processing_time_compression=t_compress_end - t_compress_start,
            processing_time_pointcloud=0, 
            processing_time_total=time.perf_counter() - t_start_total,
            point_cloud=[], error_message="Missing original image dimensions for PC conversion"
        )

    point_cloud_data = depth_to_point_cloud(
        depth_data=depth_for_pointcloud_conversion,
        camera_intrinsics=camera_intrinsics,
        is_grid_data=is_input_grid_data_for_pc,
        grid_config=grid_params_for_pc_func,
        original_image_dims=original_image_dims_tuple # Pass tuple or None
    )
    t_pc_end = time.perf_counter()

    pc_list = []
    if point_cloud_data is None or point_cloud_data.size == 0:
        logger.info("Point cloud generation resulted in no points or failed.")
    else:
        pc_list = point_cloud_data.tolist()
        logger.info(f"Generated point cloud with {len(pc_list)} points.")

    t_end_total = time.perf_counter()
    timestamp_processed = time.time()

    return PointCloudResponse(
        timestamp_capture=timestamp_capture,
        timestamp_processed=timestamp_processed,
        processing_time_depth=round(t_depth_end - t_depth_start, 4),
        processing_time_compression=round(t_compress_end - t_compress_start, 4), 
        processing_time_pointcloud=round(t_pc_end - t_pc_start, 4),
        processing_time_total=round(t_end_total - t_start_total, 4),
        point_cloud=pc_list
    )

# --- Main Execution (for running with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    
    loaded_config_for_main = load_config(CONFIG_PATH) 
    api_server_config = {}
    if loaded_config_for_main and "api_server" in loaded_config_for_main:
        api_server_config = loaded_config_for_main["api_server"]
    else:
        logger.warning("api_server configuration not found or config file failed to load. Using uvicorn defaults.")

    host = api_server_config.get("host", "0.0.0.0")
    port = api_server_config.get("port", 8000)
    
    uvicorn_log_level = logging.getLevelName(logger.getEffectiveLevel()).lower()
    if uvicorn_log_level == "notset": uvicorn_log_level = "info"


    logger.info(f"Starting Uvicorn server on http://{host}:{port}")
    uvicorn.run("linux_main:app", host=host, port=port, log_level=uvicorn_log_level, reload=True)
