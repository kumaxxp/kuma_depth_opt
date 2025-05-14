# トップダウンビューテストツール

このディレクトリには、圧縮深度データからトップダウン占有グリッドを生成するためのテストツールが含まれています。
カメラ入力なしで圧縮深度データ（CSVファイル）から直接トップダウンビューを生成し、パラメータの調整や結果の可視化を行うことができます。

## ファイル構成

- `test_top_view.py`: 基本的なトップダウンビューのテストプログラム
- `generate_test_data.py`: 異なるシナリオのテスト用CSVデータを生成
- `compare_parameters.py`: 異なるパラメータ設定での結果を比較

## 基本的な使い方

### 1. テスト用データの生成

まず、テスト用のCSVデータを生成します。以下のコマンドで、異なるシナリオの深度データが `test_data` ディレクトリに生成されます：

```bash
python generate_test_data.py --output-dir test_data
```

生成されるシナリオ：
- 空の部屋 (`empty_room_grid.csv`)
- 障害物のある部屋 (`obstacle_grid.csv`)
- 廊下 (`corridor_grid.csv`)
- 階段 (`stairs_grid.csv`)
- 複雑なシーン (`complex_scene_grid.csv`)

これらは標準解像度(12x16)と高解像度(24x32)の両方で生成されます。

### 2. 基本的なトップダウンビューのテスト

生成したCSVデータを使用してトップダウンビューをテストします：

```bash
python test_top_view.py --csv test_data/obstacle_grid.csv
```

CSVファイルを指定しない場合は、テスト用のCSVが自動生成されます：

```bash
python test_top_view.py
```

結果を保存する場合：

```bash
python test_top_view.py --csv test_data/obstacle_grid.csv --save-dir results
```

### 3. パラメータ比較テスト

1つのCSVファイルに対して異なるパラメータ設定（グリッド解像度、高さ閾値など）の結果を比較できます：

```bash
python compare_parameters.py --csv test_data/obstacle_grid.csv
```

結果を保存する場合：

```bash
python compare_parameters.py --csv test_data/obstacle_grid.csv --save-dir parameter_comparison
```

## パラメータの説明

トップダウンビュー生成に影響する主なパラメータ：

1. **grid_resolution** (メートル/セル): グリッドの解像度。小さいほど細かいグリッドになります。
   - 推奨範囲: 0.05 ~ 0.2

2. **grid_width**, **grid_height** (セル数): グリッドの幅と高さ。
   - 推奨: 解像度に応じて調整（例: 4m÷解像度）

3. **height_threshold** (メートル): 床と障害物を区別する高さの閾値。
   - 推奨範囲: 0.1 ~ 0.3

4. **scaling_factor** (深度スケール): 相対深度値から絶対深度（メートル）への変換係数。
   - 推奨範囲: 5.0 ~ 15.0

## 結果の解釈

テストプログラムでは、以下の情報が出力されます：

1. 深度グリッドの可視化: CSVデータの視覚的表現
2. 点群の上面図: 3D点群をX-Z平面に投影したもの（色は高さY）
3. 占有グリッド: トップダウンビューの最終結果
   - 灰色: 未知領域
   - 赤色: 障害物
   - 緑色: 通行可能領域

## カスタマイズ

各スクリプト内の関数やパラメータを修正することで、さらに詳細な調整が可能です。特に重要なのは以下のセクション：

- `test_top_view.py` の `process_depth_grid` 関数内のパラメータ
- `compare_parameters.py` の `grid_resolutions` と `height_thresholds` の値

## 注意事項

- テスト用CSVデータは実際のカメラデータとは異なる場合があります
- より正確なテストには、実際のカメラから取得した圧縮深度データを使用してください
