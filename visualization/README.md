# 障害物検知システム - 可視化リソース

このディレクトリには、障害物検知システムのアルゴリズムと座標系を理解するための包括的な可視化が含まれています。

## 可視化の実行

```bash
python coordinate_systems.py
```

## 生成される可視化

### 1. 包括的統合図

**`comprehensive_visualization.png`** - 4パネル統合図
- 左上: カメラ座標系とピンホールモデル
- 右上: 深度変換アルゴリズム
- 左下: グリッド圧縮マッピング  
- 右下: トップダウン変換

### 2. 個別詳細図

- **`camera_coordinate_system.png`** - カメラ座標系（X:右、Y:下、Z:前）とピンホールカメラモデル
- **`depth_conversion_algorithm.png`** - 相対深度→絶対深度変換（パーセンタイル正規化）
- **`grid_compression_mapping.png`** - 画像解像度からグリッドへの圧縮プロセス
- **`topdown_transformation.png`** - 3Dポイントクラウドから2D占有グリッドへの変換

### 3. パラメータ比較図

- **`custom_vis_s5.0_g4x3_r0.5_h0.2.png`** - 低解像度設定（4x3グリッド、0.5m解像度）
- **`custom_vis_s10.0_g8x6_r0.3_h0.15.png`** - 中解像度設定（8x6グリッド、0.3m解像度）
- **`custom_vis_s15.0_g16x12_r0.25_h0.15.png`** - 高解像度設定（16x12グリッド、0.25m解像度）

### 4. テスト・検証ツール

#### スクリプト
- **`coordinate_systems.py`** - 統合可視化生成スクリプト（メイン）
- **`test_camera_view_fixed.py`** - カメラ座標系単独テスト・検証ツール

#### 生成されるテスト結果図
- **`test_camera_coordinate_system_fixed.png`** - カメラ座標系テスト結果

## 座標系の重要なポイント

### カメラ座標系
- **X軸**: 右方向（画像の左→右）
- **Y軸**: 下方向（画像の上→下）
- **Z軸**: 前方向（カメラから物体への方向）

### 主な変換プロセス
1. **3D世界座標** → **カメラ座標** （ピンホールモデル）
2. **カメラ座標** → **画像座標** （投影）
3. **画像座標** → **グリッド座標** （圧縮）
4. **3Dポイントクラウド** → **2D占有グリッド** （トップダウン投影）

## アルゴリズムの詳細

詳細な数学的定義とアルゴリズム仕様については、`../障害物検知システムアルゴリズム.md` を参照してください。

## カスタム可視化

`coordinate_systems.py`を直接編集することで、以下をカスタマイズできます：
- スケーリング係数（scaling_factor）
- グリッドサイズ（grid_cols, grid_rows）
- 占有グリッド解像度（grid_resolution）
- 高さ閾値（height_threshold）

例：
```python
create_custom_visualization(
    scaling_factor=20.0,     # 深度スケール
    grid_cols=32, grid_rows=24,  # 高解像度グリッド
    grid_resolution=0.1,     # 10cm解像度
    height_threshold=0.1     # 10cm高さ閾値
)
```

これらの可視化により、システムの各段階での座標変換とアルゴリズムの動作を直感的に理解できます。
