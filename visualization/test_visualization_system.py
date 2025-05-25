#!/usr/bin/env python3
"""
可視化システムの包括的なテストスクリプト
すべての機能が正常に動作することを確認
"""

import matplotlib
matplotlib.use('Agg')  # バックグラウンドモード

from coordinate_systems import (
    create_comprehensive_algorithm_visualization,
    create_individual_visualizations,
    create_custom_visualization
)
import os

def test_visualization_system():
    """可視化システムの包括的テスト"""
    
    print("=== 可視化システムテスト開始 ===\n")
    
    # 1. 包括的可視化のテスト
    print("1. 包括的可視化の生成をテスト...")
    try:
        create_comprehensive_algorithm_visualization()
        if os.path.exists('comprehensive_visualization.png'):
            print("   ✓ 包括的可視化が正常に生成されました")
        else:
            print("   ✗ 包括的可視化ファイルが見つかりません")
    except Exception as e:
        print(f"   ✗ 包括的可視化でエラーが発生: {e}")
    
    # 2. 個別可視化のテスト
    print("\n2. 個別可視化の生成をテスト...")
    try:
        create_individual_visualizations()
        individual_files = [
            'camera_coordinate_system.png',
            'depth_conversion_algorithm.png', 
            'grid_compression_mapping.png',
            'topdown_transformation.png'
        ]
        
        all_files_exist = True
        for filename in individual_files:
            if os.path.exists(filename):
                print(f"   ✓ {filename} が正常に生成されました")
            else:
                print(f"   ✗ {filename} が見つかりません")
                all_files_exist = False
        
        if all_files_exist:
            print("   ✓ すべての個別可視化が正常に生成されました")
    except Exception as e:
        print(f"   ✗ 個別可視化でエラーが発生: {e}")
    
    # 3. カスタム可視化のテスト（複数のパラメータセット）
    print("\n3. カスタム可視化の生成をテスト...")
    
    test_configs = [
        {"scaling_factor": 5.0, "grid_cols": 4, "grid_rows": 3, "grid_resolution": 0.5, "height_threshold": 0.2},
        {"scaling_factor": 10.0, "grid_cols": 8, "grid_rows": 6, "grid_resolution": 0.3, "height_threshold": 0.15},
        {"scaling_factor": 15.0, "grid_cols": 16, "grid_rows": 12, "grid_resolution": 0.25, "height_threshold": 0.1}
    ]
    
    for i, config in enumerate(test_configs, 1):
        try:
            create_custom_visualization(**config)
            expected_filename = f'custom_vis_s{config["scaling_factor"]}_g{config["grid_cols"]}x{config["grid_rows"]}_r{config["grid_resolution"]}_h{config["height_threshold"]}.png'
            
            if os.path.exists(expected_filename):
                print(f"   ✓ カスタム設定 {i}: {expected_filename} が正常に生成されました")
            else:
                print(f"   ✗ カスタム設定 {i}: {expected_filename} が見つかりません")
        except Exception as e:
            print(f"   ✗ カスタム設定 {i} でエラーが発生: {e}")
    
    # 4. ファイルサイズとアクセス性の確認
    print("\n4. 生成されたファイルの品質確認...")
    
    all_files = [
        'comprehensive_visualization.png',
        'camera_coordinate_system.png',
        'depth_conversion_algorithm.png', 
        'grid_compression_mapping.png',
        'topdown_transformation.png'
    ]
    
    for filename in all_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            if file_size > 1024:  # 1KB以上
                print(f"   ✓ {filename}: {file_size:,} bytes")
            else:
                print(f"   ⚠ {filename}: ファイルサイズが小さすぎます ({file_size} bytes)")
        else:
            print(f"   ✗ {filename}: ファイルが存在しません")
    
    print("\n=== 可視化システムテスト完了 ===")
    print("\n📊 生成された可視化ファイル:")
    print("  • comprehensive_visualization.png - 4パネル包括可視化")
    print("  • camera_coordinate_system.png - カメラ座標系")
    print("  • depth_conversion_algorithm.png - 深度変換アルゴリズム")
    print("  • grid_compression_mapping.png - グリッド圧縮マッピング")
    print("  • topdown_transformation.png - トップダウン変換")
    print("  • custom_vis_*.png - カスタムパラメータ可視化")

if __name__ == "__main__":
    test_visualization_system()
