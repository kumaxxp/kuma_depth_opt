"""
深度推定モデル関連の処理
"""

import cv2
import numpy as np
import time
import os
import logging

# ロガーの取得
logger = logging.getLogger("kuma_depth_opt.depth_model")
# --- ここから追加 ---
# logger のレベルを DEBUG に設定
logger.setLevel(logging.DEBUG)
# ハンドラが設定されていなければ、標準出力へのハンドラを追加
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# --- ここまで追加 ---

# axengine をインポート
try:
    import axengine as axe
    HAS_AXENGINE = True
except ImportError:
    HAS_AXENGINE = False
    logger.warning("axengine is not installed. Running in basic mode without depth estimation.")

class DepthProcessor:
    """深度推定処理クラス"""
    
    def __init__(self, config: dict):
        """
        初期化
        
        Args:
            config: 設定辞書 (config_linux.json の内容)
        """
        self.config = config
        # Use the correct key from config_linux.json: "depth_model" -> "model_path"
        depth_model_config_main = self.config.get("depth_model", {})
        self.model_path = depth_model_config_main.get("model_path") # CORRECTED KEY

        if not self.model_path:
            logger.error("'model_path' not defined in the 'depth_model' section of the configuration.")
            # Consider raising an error or using a hardcoded default if critical
            # raise ValueError("Model path not configured")

        self.model = self._initialize_model()
        self.input_name = None
        if self.model:
            try:
                model_inputs = self.model.get_inputs()
                if model_inputs and len(model_inputs) > 0:
                    self.input_name = model_inputs[0].name
                else:
                    logger.error("Failed to get model inputs or inputs are empty.")
                    self.model = None # Invalidate model
            except Exception as e:
                logger.error(f"Failed to get model input name: {e}")
                self.model = None # Invalidate model

        default_input_width = 384
        default_input_height = 256
        
        depth_model_config = self.config.get("depth_model_parameters", {})
        self.model_input_width = depth_model_config.get("input_width", default_input_width)
        self.model_input_height = depth_model_config.get("input_height", default_input_height)
        self.target_size = (self.model_input_width, self.model_input_height) # OpenCV uses (width, height)
            
    def _initialize_model(self):
        """モデルを初期化"""
        if not HAS_AXENGINE:
            logger.warning("axengine not installed. Cannot initialize depth model.")
            return None
            
        if not self.model_path: # Check if model_path was successfully retrieved
            logger.error("Cannot initialize model: model_path is not set.")
            return None

        try:
            logger.info(f"Loading model from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return None
                 
            try:
                session = axe.InferenceSession(self.model_path)
                logger.info("Model session created successfully")
                
                try:
                    inputs = session.get_inputs()
                    if inputs and len(inputs) > 0:
                        logger.info(f"Model has {len(inputs)} inputs")
                        logger.info(f"First input name: {inputs[0].name}")
                    else:
                        logger.warning("Model has no inputs or get_inputs() returned empty.")
                except Exception as e:
                    logger.warning(f"Could not get input details: {e}, but continuing")
                    
                logger.info("Model loaded successfully")
                return session
            except Exception as e:
                logger.error(f"Failed to create inference session: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
        except Exception as e:
            logger.error(f"Failed to initialize depth model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def process_frame(self, frame): # target_size argument removed
        """
        フレームを処理用に前処理
        
        Args:
            frame: 入力画像
            
        Returns:
            前処理済みの画像テンソル
        """
        if frame is None:
            logger.error("Input frame is None in process_frame.")
            raise ValueError("フレームの読み込みに失敗しました")
        
        resized_frame = cv2.resize(frame, self.target_size) 
        return np.expand_dims(resized_frame[..., ::-1], axis=0)
        
    def predict(self, frame):
        """深度推定を実行"""
        if not self.is_available():
            logger.info("Using dummy depth data (model not available or not initialized correctly)")
            return self._generate_dummy_depth(size=(self.model_input_height, self.model_input_width)), 0.01
            
        start_time = time.time()
        
        try:
            input_tensor = self.process_frame(frame)
            logger.debug(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            if self.input_name is None: 
                logger.error("Model input name is not set. Cannot run inference.")
                return self._generate_dummy_depth(size=(self.model_input_height, self.model_input_width)), time.time() - start_time

            outputs = self.model.run(None, {self.input_name: input_tensor})
            if outputs is None or len(outputs) == 0:
                logger.error("Inference returned empty outputs")
                return self._generate_dummy_depth(size=(self.model_input_height, self.model_inputWidth)), time.time() - start_time
                
            depth = outputs[0]
            logger.debug(f"Raw depth output shape: {depth.shape}, size: {depth.size}")
            
            try:
                expected_elements = self.model_input_height * self.model_input_width
                if depth.size == expected_elements:
                    depth_map = depth.reshape(1, self.model_input_height, self.model_input_width, 1)
                else:
                    logger.warning(f"Unexpected depth output size: {depth.size}, expected: {expected_elements}. Attempting to reshape to configured HxW: {self.model_input_height}x{self.model_input_width}.")
                    # This might fail if the total number of elements doesn't match.
                    # Consider how to handle this case more robustly if it occurs.
                    depth_map = depth.reshape(1, self.model_input_height, self.model_input_width, 1) 
                    
                depth_map = np.ascontiguousarray(depth_map)
                
                min_val = np.min(depth_map)
                max_val = np.max(depth_map)
                logger.debug(f"Depth range (model output): {min_val:.4f} to {max_val:.4f}")
                
                if np.isnan(depth_map).any() or np.isinf(depth_map).any():
                    logger.warning("Depth map contains NaN or Inf values. Clamping them.")
                    depth_map = np.nan_to_num(depth_map, nan=0.5, posinf=1.0, neginf=0.0)
                
                inference_time = time.time() - start_time
                return depth_map, inference_time
                
            except Exception as e:
                logger.error(f"Error in depth post-processing (reshape/normalize): {e}")
                return self._generate_dummy_depth(size=(self.model_input_height, self.model_input_width)), time.time() - start_time
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_dummy_depth(size=(self.model_input_height, self.model_input_width)), time.time() - start_time
    
    def _generate_dummy_depth(self, size=(256, 384)): # size is (height, width)
        """テスト用のダミー深度マップを生成"""
        output_h, output_w = size 
        
        dummy_depth = np.zeros((1, output_h, output_w, 1), dtype=np.float32)
        for y_idx in range(output_h):
            value = 0.1 + 0.8 * (y_idx / output_h)
            dummy_depth[0, y_idx, :, 0] = value
            
        logger.debug(f"Generated dummy depth map with shape: {dummy_depth.shape}")
        return dummy_depth
    
    def is_available(self):
        """モデルが利用可能かどうかを返す"""
        return self.model is not None and self.input_name is not None

    def compress_depth_to_grid(self, depth_map_abs, grid_config: dict):
        """
        絶対深度マップを指定されたグリッドサイズに圧縮します。
        
        Args:
            depth_map_abs (numpy.ndarray): 絶対深度マップ (メートル単位)。
            grid_config (dict): グリッド圧縮の設定。
                                 'target_rows', 'target_cols', 'method' を含む。
        
        Returns:
            numpy.ndarray: 圧縮されたグリッド深度データ。
                           エラー時は None を返す。
        """
        if depth_map_abs is None or depth_map_abs.size == 0:
            logger.error("Input depth_map_abs is empty for grid compression.")
            return None

        target_rows = grid_config.get("target_rows")
        target_cols = grid_config.get("target_cols")
        method = grid_config.get("method", "mean")

        if not target_rows or not target_cols:
            logger.error("target_rows or target_cols not defined in grid_compressor config.") # Corrected from grid_compressor to grid_config
            return None

        original_height, original_width = depth_map_abs.shape[:2]
        cell_height = original_height // target_rows
        cell_width = original_width // target_cols

        # Initialize compressed grid
        compressed_grid = np.zeros((target_rows, target_cols), dtype=np.float32)

        for r in range(target_rows):
            for c in range(target_cols):
                # Define the region of interest in the original depth map
                roi = depth_map_abs[
                    r * cell_height : (r + 1) * cell_height,
                    c * cell_width : (c + 1) * cell_width,
                ]

                if method == "mean":
                    # Use mean of the ROI as the grid cell value
                    compressed_grid[r, c] = np.mean(roi)
                elif method == "max":
                    # Use max of the ROI as the grid cell value
                    compressed_grid[r, c] = np.max(roi)
                elif method == "min":
                    # Use min of the ROI as the grid cell value
                    compressed_grid[r, c] = np.min(roi)
                else:
                    logger.warning(f"Unknown compression method: {method}. Defaulting to mean.")
                    compressed_grid[r, c] = np.mean(roi)

        return compressed_grid

def initialize_depth_model(config: dict):
    """
    深度推定モデルを初期化する便利関数
    
    Args:
        config: 設定辞書
        
    Returns:
        DepthProcessor インスタンス
    """
    return DepthProcessor(config)

def convert_to_absolute_depth(depth_map, config: dict, is_compressed_grid: bool):
    """
    相対深度マップを絶対深度マップ（メートル単位）に変換します
    圧縮グリッドデータにも対応した最適化版
    
    Args:
        depth_map (numpy.ndarray): 相対深度マップ（0-1の範囲）
        config (dict): 設定辞書 (config_linux.json の内容)
        is_compressed_grid (bool): 圧縮グリッドデータかどうか
        
    Returns:
        numpy.ndarray: 絶対深度マップ（メートル単位）
    """
    try:
        depth_processing_config = config.get("depth_processing", {})
        scaling_factor = depth_processing_config.get("scaling_factor", 15.0) 
        depth_scale = depth_processing_config.get("depth_scale", 1.0)       

        min_val_str = f"{np.min(depth_map):.4f}" if depth_map.size > 0 else "N/A"
        max_val_str = f"{np.max(depth_map):.4f}" if depth_map.size > 0 else "N/A"
        logger.debug(f"[AbsDepth] Input depth_map shape: {depth_map.shape}, min: {min_val_str}, max: {max_val_str}")
        logger.debug(f"[AbsDepth] Using scaling_factor={scaling_factor:.2f}, depth_scale={depth_scale:.2f}, is_compressed_grid={is_compressed_grid}")
        
        if depth_map is None or depth_map.size == 0:
            logger.warning("[AbsDepth] Error: Empty depth map provided.")
            fb_shape = (12,16) # Default fallback
            if is_compressed_grid:
                grid_conf = config.get("grid_compressor", {})
                fb_rows = grid_conf.get("target_rows", 12)
                fb_cols = grid_conf.get("target_cols", 16)
                fb_shape = (fb_rows, fb_cols)
            else:
                model_params_conf = config.get("depth_model_parameters", {})
                fb_h = model_params_conf.get("input_height", 256)
                fb_w = model_params_conf.get("input_width", 384)
                fb_shape = (fb_h, fb_w)
            return np.ones(fb_shape, dtype=np.float32) * 3.0
        
        valid_mask = depth_map > 0.01 
        
        valid_count = np.sum(valid_mask)
        if depth_map.size > 0: # Should always be true if we passed the earlier check
            valid_percentage = valid_count * 100.0 / depth_map.size
            logger.info(f"[AbsDepth] Valid depth values (raw > 0.01): {valid_count}/{depth_map.size} ({valid_percentage:.1f}%)")
        else: # This case should ideally not be reached due to earlier depth_map.size == 0 check
            logger.warning("[AbsDepth] depth_map.size is 0 unexpectedly after initial checks.")
            return np.array([], dtype=np.float32)


        if valid_count == 0:
            logger.warning("[AbsDepth] Warning: No valid depth values found after initial check (all <= 0.01). Returning default depth.")
            return np.ones_like(depth_map) * 3.0
        
        absolute_depth = np.ones_like(depth_map) * 3.0 
        valid_values = depth_map[valid_mask]
        
        if valid_values.size == 0: 
             logger.warning("[AbsDepth] No values in valid_values array despite valid_count > 0. This is unexpected. Returning default depth.")
             return np.ones_like(depth_map) * 3.0

        near_point = np.max(valid_values)
        far_point = np.min(valid_values)
        median_point = np.median(valid_values)
        logger.info(f"[AbsDepth] Raw valid depth stats - Near (max_val): {near_point:.4f}, Far (min_val): {far_point:.4f}, Median: {median_point:.4f}")
        
        percentiles = np.percentile(valid_values, [5, 25, 50, 75, 95])
        logger.debug(f"[AbsDepth] Raw valid depth percentiles [5,25,50,75,95]: {percentiles}")
        
        if is_compressed_grid:
            effective_near = percentiles[4]  # 95th percentile for near
            effective_far = percentiles[0]   # 5th percentile for far
            logger.debug(f"[AbsDepth] Using percentiles for compressed data - Effective Near: {effective_near:.4f}, Effective Far: {effective_far:.4f}")
        else:
            effective_near = near_point
            effective_far = far_point
        
        depth_range = effective_near - effective_far
        if depth_range < 0.01: 
            logger.warning(f"[AbsDepth] Warning: Depth range is very small ({depth_range:.4f}). Clamping to 0.01 to avoid division by zero or instability.")
            depth_range = 0.01 
        
        diff = effective_near - depth_map 
        normalized_diff = np.clip(diff / depth_range, 0, 1) 
        
        if is_compressed_grid:
            final_scaling_factor = scaling_factor
        else:
            final_scaling_factor = scaling_factor * depth_scale 
        
        absolute_depth[valid_mask] = 0.5 + normalized_diff[valid_mask] * final_scaling_factor
        absolute_depth = np.clip(absolute_depth, 0.1, 50.0) # Clamp to 0.1m to 50m

        # Log final stats
        final_valid_mask_for_stats = (absolute_depth >= 0.1) & (absolute_depth <= 50.0)
        if np.any(final_valid_mask_for_stats): # Check if any values are in the valid range
            valid_abs_depths = absolute_depth[final_valid_mask_for_stats]
            if valid_abs_depths.size > 0: # Ensure array is not empty
                 abs_min = np.min(valid_abs_depths)
                 abs_max = np.max(valid_abs_depths)
                 abs_mean = np.mean(valid_abs_depths)
                 logger.info(f"[AbsDepth] Final absolute depth stats (0.1m-50m range) - Min: {abs_min:.2f}m, Max: {abs_max:.2f}m, Mean: {abs_mean:.2f}m")
            else:
                logger.info("[AbsDepth] No valid absolute depth values in the final 0.1m-50m range after filtering for stats.")
        else:
            logger.info("[AbsDepth] No valid absolute depth values in the final 0.1m-50m range.")
        
        return absolute_depth
        
    except Exception as e:
        logger.error(f"[AbsDepth] Error in convert_to_absolute_depth: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
        fb_shape = (12,16) # Default fallback
        try: # Try to determine fallback shape from config if possible
            if depth_map is not None and hasattr(depth_map, 'shape') and depth_map.size > 0 : # Check if depth_map has shape
                 fb_shape = depth_map.shape
            elif config: # if config is available
                if is_compressed_grid:
                    grid_conf = config.get("grid_compressor", {})
                    fb_rows = grid_conf.get("target_rows", 12)
                    fb_cols = grid_conf.get("target_cols", 16)
                    fb_shape = (fb_rows, fb_cols)
                else:
                    model_params_conf = config.get("depth_model_parameters", {})
                    fb_h = model_params_conf.get("input_height", 256)
                    fb_w = model_params_conf.get("input_width", 384)
                    fb_shape = (fb_h, fb_w)
        except Exception as shape_ex:
            logger.error(f"[AbsDepth] Error determining fallback shape: {shape_ex}. Using default (12,16).")
            fb_shape = (12,16)
            
        return np.ones(fb_shape, dtype=np.float32) * 3.0