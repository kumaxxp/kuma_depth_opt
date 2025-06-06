"""
Depth map visualization functions
"""

import cv2
import numpy as np
import logging

# Import English text utilities
from english_text_utils import setup_matplotlib_english, cv2_put_english_text

# Get logger
logger = logging.getLogger("kuma_depth_opt.visualization")
# --- Start of addition ---
# Set logger level to DEBUG
logger.setLevel(logging.DEBUG)
# Add handler to standard output if no handlers are set
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.info("Logger for 'kuma_depth_opt.visualization' INITIALIZED. This message should appear if the module is imported and logger is working.")
# --- End of addition ---

def create_depth_visualization(depth_map, original_shape, add_colorbar=True):
    """Visualize depth map"""
    try:
        if depth_map is None or depth_map.size == 0:
            logger.warning("Empty depth map received for visualization")
            return create_default_depth_image(
                640 if original_shape is None else original_shape[1],
                480 if original_shape is None else original_shape[0]
            )
            
        # Log depth map shape
        logger.debug(f"Visualizing depth map with shape: {depth_map.shape}")
        
        # Convert depth map to 2D
        if len(depth_map.shape) == 4:  # (1, H, W, 1) format
            depth_feature = depth_map.reshape(depth_map.shape[1:3])
        elif len(depth_map.shape) == 3:  # (H, W, 1) or (1, H, W) format
            if depth_map.shape[2] == 1:
                depth_feature = depth_map.reshape(depth_map.shape[:2])
            else:
                depth_feature = depth_map.reshape(depth_map.shape[1:])
        else:
            depth_feature = depth_map  # already 2D
            
        # NaNやInfをチェックして置換
        depth_feature = np.nan_to_num(depth_feature, nan=0.5, posinf=1.0, neginf=0.1)
        
        # 値の範囲を確認
        logger.debug(f"Depth feature range: {np.min(depth_feature):.4f} to {np.max(depth_feature):.4f}")
        
        # 深度の正規化（無効値を除外）
        valid_depth = depth_feature[depth_feature > 0.01]
        if len(valid_depth) > 0:
            min_depth = np.percentile(valid_depth, 5)  # 外れ値を除外
            max_depth = np.percentile(valid_depth, 95) # 外れ値を除外
        else:
            logger.warning("No valid depth values found")
            min_depth = 0.1
            max_depth = 0.9
            
        logger.debug(f"Using depth range for normalization: {min_depth:.4f} to {max_depth:.4f}")
        
        # 正規化して0-1範囲にする
        normalized = np.zeros_like(depth_feature, dtype=np.float32)
        valid_mask = depth_feature > 0.01
        if np.any(valid_mask) and (max_depth > min_depth):
            normalized[valid_mask] = np.clip(
                (depth_feature[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6), 
                0, 1
            )
            
        # colormap適用
        depth_colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_MAGMA
        )
        
        # 元の画像サイズにリサイズ
        if original_shape is not None and len(original_shape) >= 2:
            return cv2.resize(depth_colored, (original_shape[1], original_shape[0]))
            
        return depth_colored
            
    except Exception as e:
        logger.error(f"Error in create_depth_visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # エラー時はデフォルト画像を返す
        return create_default_depth_image(
            640 if original_shape is None else original_shape[1], 
            480 if original_shape is None else original_shape[0]
        )

def create_default_depth_image(width=640, height=480, text=None):
    """
    デフォルトの深度イメージを生成（モデルがない場合のプレースホルダ）
    
    Args:
        width: 画像幅
        height: 画像高さ
        text: 表示するテキスト（Noneの場合はテキストなし）
        
    Returns:
        デフォルトの深度イメージ
    """
    default_depth_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # グラデーション背景を生成
    for i in range(height):
        default_depth_image[i, :] = [0, 0, int(255 * i / height)]
    
    # テキストを表示（指定された場合）
    if text is not None:
        # 画像サイズに応じてフォントサイズを調整
        font_scale = min(width, height) / 500.0  # サイズに応じてスケール調整
        font_scale = max(0.5, min(font_scale, 1.5))  # 0.5から1.5の間に制限
        thickness = max(1, int(font_scale * 2))
        
        # テキストが長い場合は複数行に分割
        max_text_width = int(width * 0.9)  # 画像幅の90%を最大幅とする
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        
        # 日本語対応のためのチェック
        has_japanese = any(ord(c) > 127 for c in text)
        
        if has_japanese:
            try:
                # fix_text_encoding.pyのcv2_putText_ja関数を使用
                from fix_text_encoding import cv2_putText_ja
                
                # テキストが長い場合は複数行に分割
                if len(text) > width // 10:  # 簡易的な判断
                    mid_point = len(text) // 2
                    # スペースがあればそこで分割
                    for i in range(mid_point, min(len(text), mid_point + 20)):
                        if text[i] == ' ':
                            mid_point = i
                            break
                    
                    text1 = text[:mid_point]
                    text2 = text[mid_point:].strip()
                    
                    # テキスト位置を計算
                    text1_x = width // 2 - len(text1) * int(font_scale * 10) // 2
                    text1_y = height // 2 - int(font_scale * 15)
                    
                    text2_x = width // 2 - len(text2) * int(font_scale * 10) // 2
                    text2_y = height // 2 + int(font_scale * 25)
                    
                    # 各行を描画
                    default_depth_image = cv2_putText_ja(default_depth_image, text1, (text1_x, text1_y), 
                                                       font_face, font_scale, (255, 255, 255), thickness)
                    default_depth_image = cv2_putText_ja(default_depth_image, text2, (text2_x, text2_y), 
                                                       font_face, font_scale, (255, 255, 255), thickness)
                else:
                    # 1行で十分な場合
                    text_x = width // 2 - len(text) * int(font_scale * 10) // 2
                    text_y = height // 2 + int(font_scale * 10)
                    
                    default_depth_image = cv2_putText_ja(default_depth_image, text, (text_x, text_y), 
                                                      font_face, font_scale, (255, 255, 255), thickness)
                
            except ImportError:
                logger.warning("fix_text_encoding module not found. Japanese text may not display correctly.")
                # 通常のcv2.putTextにフォールバック
                # テキストの分割が必要か確認
                (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
                
                if text_width > max_text_width:
                    # 長いテキストを分割
                    mid_point = len(text) // 2
                    text1 = text[:mid_point]
                    text2 = text[mid_point:].strip()
                    
                    (text1_width, text1_height), _ = cv2.getTextSize(text1, font_face, font_scale, thickness)
                    
                    # テキスト位置を計算して描画
                    text1_x = (width - text1_width) // 2
                    text1_y = (height // 2) - int(text1_height * 0.5)
                    cv2.putText(default_depth_image, text1, (text1_x, text1_y), 
                               font_face, font_scale, (255, 255, 255), thickness)
                    
                    # 2行目
                    (text2_width, text2_height), _ = cv2.getTextSize(text2, font_face, font_scale, thickness)
                    text2_x = (width - text2_width) // 2
                    text2_y = (height // 2) + int(text2_height * 1.5)
                    cv2.putText(default_depth_image, text2, (text2_x, text2_y), 
                               font_face, font_scale, (255, 255, 255), thickness)
                else:
                    # 1行で十分な場合
                    text_x = (width - text_width) // 2
                    text_y = (height + text_height) // 2
                    cv2.putText(default_depth_image, text, (text_x, text_y), 
                               font_face, font_scale, (255, 255, 255), thickness)
        else:
            # 非日本語文字の場合は通常のcv2.putTextを使用
            # テキストの分割が必要か確認
            (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
            
            if text_width > max_text_width:
                # 長いテキストは複数行に分割（単純に半分で区切る）
                mid_point = len(text) // 2
                # スペースがあればそこで分割
                for i in range(mid_point, min(len(text), mid_point + 20)):
                    if text[i] == ' ':
                        mid_point = i
                        break
                
                text1 = text[:mid_point]
                text2 = text[mid_point:].strip()
                
                (text1_width, text1_height), _ = cv2.getTextSize(text1, font_face, font_scale, thickness)
                (text2_width, text2_height), _ = cv2.getTextSize(text2, font_face, font_scale, thickness)
                
                # 1行目のテキスト位置
                text1_x = (width - text1_width) // 2
                text1_y = (height // 2) - int(text1_height * 0.5)
                
                # 2行目のテキスト位置
                text2_x = (width - text2_width) // 2
                text2_y = (height // 2) + int(text2_height * 1.5)
                
                # テキスト描画
                cv2.putText(default_depth_image, text1, (text1_x, text1_y), 
                           font_face, font_scale, (255, 255, 255), thickness)
                cv2.putText(default_depth_image, text2, (text2_x, text2_y), 
                           font_face, font_scale, (255, 255, 255), thickness)
            else:
                # 1行で十分な場合
                # テキストの位置を計算（中央配置）
                text_x = (width - text_width) // 2
                text_y = (height + text_height) // 2
                
                # テキストを描画
                cv2.putText(default_depth_image, text, (text_x, text_y), 
                           font_face, font_scale, (255, 255, 255), thickness)
    
    return default_depth_image

def depth_to_color(depth_normalized):
    """
    深度値を色に変換（青から赤のグラデーション）
    
    Args:
        depth_normalized: 正規化された深度値（0.0〜1.0）
        
    Returns:
        色（BGR形式）
    """
    # HSV色空間での青から赤へのグラデーション
    hue = int((1.0 - depth_normalized) * 120)  # 0〜120の範囲
    saturation = 255
    value = 255
    
    # HSVからBGRへの変換
    return cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]

def create_depth_grid_visualization(depth_grid_map, absolute_depth=None, cell_size=40): # 引数と既定値変更
    """
    圧縮済みの深度グリッドマップの可視化を行う
    
    Args:
        depth_grid_map: 圧縮済みの深度グリッドマップ (rows, cols)
        absolute_depth: 絶対深度マップ（オプション、現在はグリッドセルごとの絶対深度表示に使用）
        cell_size: セルサイズ（ピクセル）- 小さくして表示を最適化

    Returns:
        グリッド可視化画像
    """
    if depth_grid_map is None or depth_grid_map.size == 0:
        logger.warning("Empty depth_grid_map received for visualization.")
        return create_default_depth_image() # デフォルト画像を返す

    rows, cols = depth_grid_map.shape # grid_size は depth_grid_map.shape から取得

    try:
        # depth_grid_map を depth_conv として使用
        depth_conv = depth_grid_map

        # 以下、可視化用の処理
        valid_depth = depth_conv[depth_conv > 0.01]
        if len(valid_depth) > 0:
            min_depth = np.percentile(valid_depth, 5)
            max_depth = np.percentile(valid_depth, 95)
        else:
            logger.warning("No valid depth values found in depth_conv for grid visualization")
            min_depth = 0.1
            max_depth = 0.9

        normalized = np.zeros_like(depth_conv, dtype=np.float32)
        valid_mask = depth_conv > 0.01
        if np.any(valid_mask) and (max_depth > min_depth):
            normalized[valid_mask] = np.clip(
                (depth_conv[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6), 
                0, 1
            )
            
        depth_colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_MAGMA
        )
        
        output = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                y_start, y_end = i * cell_size, (i + 1) * cell_size
                x_start, x_end = j * cell_size, (j + 1) * cell_size
                cell_color = depth_colored[i, j]
                output[y_start:y_end, x_start:x_end] = cell_color
                
                # テキスト表示部分を削除（深度値の表示をなくす）
                # ↓↓↓ この部分を削除 ↓↓↓
                # if absolute_depth is not None: 
                #    ...テキスト表示の処理...

        # グリッド線を描画（薄くして目立たなくする）
        grid_color = (30, 30, 30)  # より暗いグレー
        grid_thickness = 1         # 細い線幅
        
        for i in range(rows + 1):
            y = i * cell_size
            cv2.line(output, (0, y), (cols * cell_size, y), grid_color, grid_thickness)
        for j in range(cols + 1):
            x = j * cell_size
            cv2.line(output, (x, 0), (x, rows * cell_size), grid_color, grid_thickness)

        return output
        
    except Exception as e:
        logger.error(f"Error in create_depth_grid_visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return create_default_depth_image()