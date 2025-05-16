#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix text encoding issues in visualizations
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import os
from pathlib import Path
import sys

def setup_matplotlib_ja():
    """
    Configure matplotlib for proper text display in all environments
    """
    print("Configuring fonts for text display...")
      # Use a simpler approach that works with most environments
    # Use English text for all labels to ensure compatibility
    
    # Basic matplotlib configuration for better display
    matplotlib.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'axes.unicode_minus': False  # Ensure minus signs display correctly
    })
    
    # 日本語フォントの候補リスト
    ja_font_names = [
        'Yu Gothic', 'Meiryo', 'MS Gothic', 'Hiragino Kaku Gothic ProN',
        'TakaoGothic', 'Noto Sans CJK JP', 'IPAPGothic', 'VL Gothic'
    ]
    
    # フォントファミリーを探す
    font_found = False
    for font_name in ja_font_names:
        try:
            matplotlib.rc('font', family=font_name)
            # テスト用テキスト表示
            fig, ax = plt.figure(figsize=(1, 1)), plt.axes()
            ax.text(0.5, 0.5, "テスト", ha='center')
            plt.close()
            print(f"フォント '{font_name}' を使用します")
            font_found = True
            break
        except Exception:
            continue
    
    if not font_found:
        print("警告: 日本語フォントが見つかりませんでした。代替として日本語フォントファイルをダウンロードします。")
        try:
            # matplotlibで日本語フォントのダウンロードを試みる
            import matplotlib.font_manager as fm
            import shutil
            import tempfile
            import requests
            
            # フォントをダウンロード (Google Noto Sans JP)
            font_url = 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansJP-Regular.otf'
            temp_dir = tempfile.gettempdir()
            font_path = os.path.join(temp_dir, 'NotoSansJP-Regular.otf')
            
            print(f"フォントをダウンロード中: {font_url}")
            response = requests.get(font_url, stream=True)
            with open(font_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            
            # フォントキャッシュを更新
            fm.fontManager.addfont(font_path)
            matplotlib.rc('font', family='Noto Sans JP')
            print(f"ダウンロードしたフォント 'Noto Sans JP' を使用します")
        except Exception as e:
            print(f"フォントのダウンロードに失敗しました: {e}")
            print("日本語表示が正しく行われない可能性があります。")
    
    # matplotlibの設定を更新
    matplotlib.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示
    print("日本語フォント設定完了")

def cv2_putText_ja(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_AA):
    """
    OpenCVで日本語テキストを描画する関数
    
    Args:
        img: 描画対象の画像
        text: 描画するテキスト
        org: テキストの左下座標 (x, y)
        fontFace: フォント種類（cv2.FONT_HERSHEY_*）
        fontScale: フォントスケール
        color: テキストの色 (B, G, R)
        thickness: 線の太さ
        lineType: 線種類
        
    Returns:
        テキストが描画された画像
    """
    # 通常のcv2.putTextを使用して試行（ASCII文字のみ対応可能）
    if all(ord(c) < 128 for c in text):
        return cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)
    
    # 日本語文字を含む場合は代替手法を使用
    try:
        # PILを使用して日本語テキストを描画
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # 画像をPIL形式に変換
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # フォントを指定（日本語対応フォント）
        font_size = int(fontScale * 30)  # フォントサイズを調整
        
        # Windowsの場合のフォントパス
        if os.name == 'nt':
            font_paths = [
                r"C:\Windows\Fonts\meiryo.ttc",
                r"C:\Windows\Fonts\msgothic.ttc",
                r"C:\Windows\Fonts\YuGothM.ttc",
            ]
        # macOSの場合
        elif sys.platform == 'darwin':
            font_paths = [
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/AppleGothic.ttf",
            ]
        # Linuxの場合
        else:
            font_paths = [
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            ]
        
        # フォントファイルが見つかるまで試行
        font = None
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except Exception:
                continue
        
        # フォントが見つからない場合はデフォルトフォントを使用
        if font is None:
            font = ImageFont.load_default()
        
        # 文字を描画
        draw.text(org, text, font=font, fill=tuple(reversed(color)))
        
        # PIL画像をNumPy配列に戻す
        result_img = np.array(pil_img)
        return result_img
        
    except ImportError:
        print("PIL (Pillow) がインストールされていません。日本語テキストの描画には必要です。")
        print("pip install pillow でインストールしてください。")
        # 代替として、テキストを個別の文字として描画
        for i, c in enumerate(text):
            x = org[0] + i * int(fontScale * 10)
            try:
                img = cv2.putText(img, c, (x, org[1]), fontFace, fontScale, color, thickness, lineType)
            except:
                pass
        return img
    except Exception as e:
        print(f"テキスト描画中にエラーが発生しました: {e}")
        return img

# テスト用コード
if __name__ == "__main__":
    # matplotlibの設定
    setup_matplotlib_ja()
    
    # matplotlibでの日本語表示テスト
    plt.figure(figsize=(8, 4))
    plt.title("日本語表示テスト - matplotlib")
    plt.text(0.5, 0.5, "これは日本語のテストです。", ha='center', va='center', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("matplotlib_ja_test.png")
    plt.close()
    
    # OpenCVでの日本語表示テスト
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    img = cv2_putText_ja(img, "これは日本語のテストです。", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite("opencv_ja_test.png", img)
    
    print("テストが完了しました。生成された画像ファイルを確認してください。")
