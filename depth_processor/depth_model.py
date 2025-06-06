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

# デフォルトのモデルパス
DEFAULT_MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

class DepthProcessor:
    """深度推定処理クラス"""
    
    def __init__(self, model_path=None):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス。Noneの場合はデフォルトパスを使用
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = self._initialize_model()
        self.input_name = None
        if self.model:
            self.input_name = self.model.get_inputs()[0].name
            
    def _initialize_model(self):
        """モデルを初期化"""
        if not HAS_AXENGINE:
            logger.warning("axengine not installed. Cannot initialize depth model.")
            return None
            
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # ファイルの存在確認
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return None
                 
            # セッション作成 - エラー処理を強化
            try:
                session = axe.InferenceSession(self.model_path)
                logger.info("Model session created successfully")
                
                # モデル入力情報の取得は成功した場合のみ
                try:
                    inputs = session.get_inputs()
                    if inputs and len(inputs) > 0:
                        logger.info(f"Model has {len(inputs)} inputs")
                        logger.info(f"First input name: {inputs[0].name}")
                except Exception as e:
                    logger.warning(f"Could not get input details: {e}, but continuing")
                    
                # エラーがあっても、セッションは返す（推論は可能な場合があるため）
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
            
    def process_frame(self, frame, target_size=(384, 256)):
        """
        フレームを処理用に前処理
        
        Args:
            frame: 入力画像
            target_size: リサイズ後のサイズ
            
        Returns:
            前処理済みの画像テンソル
        """
        if frame is None:
            raise ValueError("フレームの読み込みに失敗しました")
        
        resized_frame = cv2.resize(frame, target_size)
        # RGB -> BGR の変換とバッチ次元の追加
        return np.expand_dims(resized_frame[..., ::-1], axis=0)
        
    def predict(self, frame):
        """深度推定を実行"""
        if not self.is_available():
            # ダミーデータを使用
            logger.info("Using dummy depth data (model not available)")
            return self._generate_dummy_depth(frame), 0.01
            
        start_time = time.time()
        
        try:
            # フレーム前処理
            input_tensor = self.process_frame(frame)
            
            # 入力テンソルの形状ログ出力
            logger.debug(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            # 推論実行
            outputs = self.model.run(None, {self.input_name: input_tensor})
            if outputs is None or len(outputs) == 0:
                logger.error("Inference returned empty outputs")
                return self._generate_dummy_depth(frame), time.time() - start_time
                
            depth = outputs[0]
            logger.debug(f"Raw depth output shape: {depth.shape}, size: {depth.size}")
            
            # 後処理 - 形状の明示的指定
            try:
                # 深度データを整形
                if depth.size == 384*256:  # 期待サイズの場合
                    depth_map = depth.reshape(1, 256, 384, 1)
                else:
                    # 形状が異なる場合、ログに出力して可能な限り調整
                    logger.warning(f"Unexpected depth size: {depth.size}, expected: {384*256}")
                    h = int(np.sqrt(depth.size / 384))
                    w = 384 if h > 0 else int(np.sqrt(depth.size))
                    h = h or 256
                    depth_map = depth.reshape(1, h, w, 1)
                    
                depth_map = np.ascontiguousarray(depth_map)
                
                # 正規化して値の範囲をチェック
                min_val = np.min(depth_map)
                max_val = np.max(depth_map)
                logger.debug(f"Depth range: {min_val:.4f} to {max_val:.4f}")
                
                # 異常値チェック（NaNやInf）
                if np.isnan(depth_map).any() or np.isinf(depth_map).any():
                    logger.warning("Depth map contains NaN or Inf values")
                    depth_map = np.nan_to_num(depth_map, nan=0.5, posinf=1.0, neginf=0.0)
                
                inference_time = time.time() - start_time
                return depth_map, inference_time
                
            except Exception as e:
                logger.error(f"Error in depth post-processing: {e}")
                return self._generate_dummy_depth(frame), time.time() - start_time
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_dummy_depth(frame), time.time() - start_time
    
    def _generate_dummy_depth(self, frame):
        """テスト用のダミー深度マップを生成"""
        if frame is None:
            h, w = 480, 640
        else:
            h, w = frame.shape[:2]
            
        # モデルの出力サイズに合わせる
        output_h, output_w = 256, 384
        
        # グラデーションパターン: 下に行くほど深度値が大きくなる（近い）
        dummy_depth = np.zeros((1, output_h, output_w, 1), dtype=np.float32)
        for y in range(output_h):
            value = 0.1 + 0.8 * (y / output_h)
            dummy_depth[0, y, :, 0] = value
            
        logger.debug(f"Generated dummy depth map: {dummy_depth.shape}")
        return dummy_depth
    
    def is_available(self):
        """モデルが利用可能かどうかを返す"""
        return self.model is not None

    def compress_depth_to_grid(self, depth_map, grid_size=(16, 12)):
        """
        深度マップを指定されたグリッドサイズに圧縮します。
        OpenCVのresizeを利用して高速化。

        Args:
            depth_map (numpy.ndarray): 入力深度マップ (H, W) または (1, H, W, 1)
            grid_size (tuple): 圧縮後のグリッドサイズ (rows, cols)

        Returns:
            numpy.ndarray: 圧縮された深度グリッド (rows, cols)
        """
        try:
            if depth_map is None or depth_map.size == 0:
                logger.warning("Cannot compress empty depth map.")
                return None

            # 深度マップを2Dに整形
            if len(depth_map.shape) == 4:  # (1, H, W, 1)
                depth_feature = depth_map.reshape(depth_map.shape[1:3])
            elif len(depth_map.shape) == 3:  # (H, W, 1)
                depth_feature = depth_map.reshape(depth_map.shape[:2])
            elif len(depth_map.shape) == 2:  # (H, W)
                depth_feature = depth_map
            else:
                logger.error(f"Unsupported depth_map shape: {depth_map.shape}")
                return None
            
            grid_rows, grid_cols = grid_size
            if grid_rows <= 0 or grid_cols <= 0:
                logger.error(f"Invalid grid_size: {grid_size}. Dimensions must be positive.")
                return None

            # NaNやInfをチェックして置換 (圧縮前に実施)
            # 無効値は0として扱い、後の処理で1e-5以下の値としてフィルタリングされる
            depth_feature = np.nan_to_num(depth_feature, nan=0.0, posinf=0.0, neginf=0.0)

            # 有効な深度値のマスクを作成 (1e-5より大きい値)
            value_mask = (depth_feature > 1e-5).astype(np.float32)
            
            # 有効な深度値のみを含むマップを作成 (他は0)
            depth_for_sum = depth_feature * value_mask

            # cv2.resizeのdsizeは (width, height) なので (grid_cols, grid_rows) を指定
            target_cv_size = (grid_cols, grid_rows)

            # 有効深度値の合計を各セルで計算 (スケーリングされた値)
            sum_resized = cv2.resize(depth_for_sum, target_cv_size, interpolation=cv2.INTER_AREA)
            
            # 有効深度値の数を各セルで計算 (スケーリングされた値)
            count_resized = cv2.resize(value_mask, target_cv_size, interpolation=cv2.INTER_AREA)

            # 平均値を計算
            compressed_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            
            # count_resizedが非常に小さい（ほぼ0）場合は除算を避ける
            # これは、元のセルに有効な深度値が全くなかった場合に相当
            valid_counts_mask = count_resized > 1e-6 # 小さな閾値

            compressed_grid[valid_counts_mask] = sum_resized[valid_counts_mask] / count_resized[valid_counts_mask]
            
            return compressed_grid

        except Exception as e:
            logger.error(f"Error in compress_depth_to_grid: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def initialize_depth_model(model_path=None):
    """
    深度推定モデルを初期化する便利関数
    
    Args:
        model_path: モデルファイルのパス
        
    Returns:
        DepthProcessor インスタンス
    """
    return DepthProcessor(model_path)

def convert_to_absolute_depth(depth_map, scaling_factor=15.0, depth_scale=1.0, is_compressed_grid=True):
    """
    相対深度マップを絶対深度マップ（メートル単位）に変換します
    圧縮グリッドデータにも対応した最適化版
    
    Args:
        depth_map (numpy.ndarray): 相対深度マップ（0-1の範囲）
        scaling_factor (float): スケーリング係数（キャリブレーションで決定）
        depth_scale (float): スケーリング補正値
        is_compressed_grid (bool): 圧縮グリッドデータかどうか
        
    Returns:
        numpy.ndarray: 絶対深度マップ（メートル単位）
    """
    try:
        logger.debug(f"[AbsDepth] Input depth_map shape: {depth_map.shape}, range: {np.min(depth_map):.4f} to {np.max(depth_map):.4f}")
        logger.debug(f"[AbsDepth] Using scaling_factor={scaling_factor:.2f}, depth_scale={depth_scale:.2f}, is_compressed_grid={is_compressed_grid}")
        
        if depth_map is None or depth_map.size == 0:
            logger.warning("[AbsDepth] Error: Empty depth map")
            # 空のデータの場合は2~5mの範囲のデフォルト深度マップを返す（テスト表示用）
            if is_compressed_grid:
                return np.ones((12, 16), dtype=np.float32) * 3.0
            else:
                return np.ones((256, 384), dtype=np.float32) * 3.0
        
        # 深度マップがゼロに近い値を持つ場所を処理（ゼロ除算防止）
        valid_mask = depth_map > 0.01
        
        # 有効なデータの割合を計算
        valid_percentage = np.sum(valid_mask) * 100.0 / depth_map.size
        valid_count = np.sum(valid_mask)
        logger.info(f"[AbsDepth] Valid depth values: {valid_count}/{depth_map.size} ({valid_percentage:.1f}%)")
        
        if valid_count == 0:
            logger.warning("[AbsDepth] Warning: No valid depth values found")
            return np.ones_like(depth_map) * 3.0  # デフォルト3メートル
        
        # 絶対深度マップの初期化
        absolute_depth = np.ones_like(depth_map) * 3.0  # デフォルト3メートル
        
        # 統計情報を出力
        valid_values = depth_map[valid_mask]
        near_point = np.max(valid_values)
        far_point = np.min(valid_values)
        median_point = np.median(valid_values)
        logger.info(f"[AbsDepth] Depth stats - Near: {near_point:.4f}, Far: {far_point:.4f}, Median: {median_point:.4f}")
        
        # パーセンタイル値も計算（異常値の影響を減らすため）
        percentiles = np.percentile(valid_values, [5, 25, 50, 75, 95])
        logger.debug(f"[AbsDepth] Depth percentiles [5,25,50,75,95]: {percentiles}")
        
        # 圧縮グリッドデータの場合は異常値除外を強化
        if is_compressed_grid:
            # 圧縮データの場合、パーセンタイルベースで計算することで誤差を削減
            effective_near = percentiles[4]  # 95パーセンタイル値を「最も近い点」として使用
            effective_far = percentiles[0]   # 5パーセンタイル値を「最も遠い点」として使用
            logger.debug(f"[AbsDepth] Using percentiles for compressed data - Near: {effective_near:.4f}, Far: {effective_far:.4f}")
        else:
            # 非圧縮データの場合、そのまま使用
            effective_near = near_point
            effective_far = far_point
        
        # 安全な計算のため範囲チェック
        depth_range = effective_near - effective_far
        if depth_range < 0.01:  # 深度範囲が小さすぎる場合
            logger.warning("[AbsDepth] Warning: Depth range too small, using default transformation")
            depth_range = 0.01  # 最小値設定
        
        # 最も近いポイントを基準にスケーリング
        # 圧縮データ用の調整されたアルゴリズム
        diff = effective_near - depth_map
        
        # 正規化して絶対深度に変換
        normalized_diff = np.clip(diff / depth_range, 0, 1) # 0-1の範囲に制限
        # スケーリング係数は圧縮データ用に調整
        adjusted_scaling = scaling_factor * (1.0 if is_compressed_grid else depth_scale)
        
        # 有効なピクセルにのみ適用
        absolute_depth[valid_mask] = 0.5 + normalized_diff[valid_mask] * adjusted_scaling
        
        # 異常な深度値をフィルタリング
        absolute_depth[absolute_depth < 0.1] = 0.1  # 最小深度
        absolute_depth[absolute_depth > 50.0] = 50.0  # 最大深度
        
        # 結果の統計情報
        valid_mask = absolute_depth > 0.1  # 有効な深度値マスク更新
        if np.any(valid_mask):
            abs_min = np.min(absolute_depth[valid_mask])
            abs_max = np.max(absolute_depth[valid_mask])
            abs_mean = np.mean(absolute_depth[valid_mask])
            logger.info(f"[AbsDepth] Absolute depth stats - Min: {abs_min:.2f}m, Max: {abs_max:.2f}m, Mean: {abs_mean:.2f}m")
        
        return absolute_depth
        
    except Exception as e:
        logger.error(f"[AbsDepth] Error in convert_to_absolute_depth: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        # エラー時はデフォルト値を返す
        return np.ones_like(depth_map) * 3.0