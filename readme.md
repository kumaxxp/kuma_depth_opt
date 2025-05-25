# Kuma Depth Opt

リアルタイム深度推定とPoint Cloud生成によるLAN経由障害物検知システムです。

## 概要

本システムは2つの実装方式を提供しています：

1. **Linux Main API** - PointCloud APIによるWindows-Linux間リアルタイム通信
2. **Simple FastAPI Camera** - ブラウザ経由のカメラストリーミングとトップダウンビュー

## 主な機能

- Depth Anythingモデルによるリアルタイム深度推定
- PointCloud生成と3D座標変換
- グリッド圧縮による処理負荷軽減
- ブラウザベースの可視化（カメラ映像、深度マップ、トップダウンビュー）
- LAN経由でのリアルタイム3Dデータ送信

## 必要条件

- Python 3.8以上
- USB カメラ（Linux Main API使用時）
- Linux環境（メインAPI用）、Windows環境（クライアント用）
- LAN接続（Windows-Linux間通信時）

## インストール方法

1. このリポジトリをクローンまたはダウンロードします。

2. 必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

3. Depth Anythingモデルをダウンロードし、指定のディレクトリに配置します（設定ファイルで指定されたパスに配置）。

## 設定ファイル

実行前に設定ファイルを適切に編集してください：

### Linux Main API用（`config_linux.json`）
```json
{
  "camera": {
    "device_id": 0,
    "width": 640,
    "height": 480,
    "fps": 30
  },
  "depth": {
    "model_path": "/path/to/depth_anything_model",
    "device": "cuda",
    "scaling_factor": 10.0
  },
  "grid": {
    "rows": 8,
    "cols": 6,
    "aggregation": "mean"
  },
  "camera_params": {
    "fx": 525.0,
    "fy": 525.0,
    "cx": 320.0,
    "cy": 240.0
  }
}
```

### Windowsクライアント用（`config_windows.json`）
```json
{
  "server": {
    "host": "192.168.1.100",
    "port": 8000
  },
  "visualization": {
    "point_size": 5,
    "color_map": "viridis",
    "update_interval": 200
  }
}
```

## 実行方法

### 方式1: Linux Main API + Windowsクライアント

**Linux側（サーバー）：**
```bash
python linux_main.py
```

**Windows側（クライアント）：**
```bash
python windows_client.py
```

APIサーバーが起動すると、以下のエンドポイントが利用可能になります：
- `GET /pointcloud` - PointCloudデータの取得
- `GET /health` - サーバー状態確認

### 方式2: Simple FastAPI Camera（単体実行）

```bash
python simple_fastapi_camera.py
```

ブラウザで以下のURLにアクセスします：
```
http://localhost:8888
```

利用可能なエンドポイント：
- `GET /` - Webインターフェース
- `GET /video` - カメラ映像ストリーム
- `GET /depth_video` - 深度マップストリーム
- `GET /depth_grid` - 深度グリッドストリーム
- `GET /top_down_view` - トップダウンビューストリーム

## 座標系について

本システムでは以下の座標系を使用しています：

### カメラ座標系
- **X軸**: 右方向が正
- **Y軸**: 下方向が正  
- **Z軸**: カメラから前方が正

### PointCloud変換
ピンホールカメラモデルを使用して、画像座標から3D座標に変換：
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = 絶対深度値
```

詳細な座標系の関係は `visualization/coordinate_systems.py` を参照してください。

## 深度値の変換について

Depth Anythingの出力は相対深度値のため、絶対距離への変換が必要です：

### 現在の実装（パーセンタイル正規化）
```python
def convert_to_absolute_depth(relative_depth, scaling_factor=10.0):
    # 5-95パーセンタイルを基準とした正規化
    p5 = np.percentile(relative_depth, 5)
    p95 = np.percentile(relative_depth, 95)
    normalized = (relative_depth - p5) / (p95 - p5)
    return normalized * scaling_factor
```

### パラメータ調整
- `scaling_factor`: 最大深度値を制御（推奨値: 5.0-15.0）
- グリッド圧縮時は元の座標系との対応を保つため座標変換を調整

## テスト用ツール

実際のカメラなしでシステムをテストするためのツールが用意されています：

```bash
# テストデータの生成
python generate_test_data.py --output-dir test_data

# トップダウンビューのテスト
python test_top_view.py --csv test_data/obstacle_grid.csv --save-dir results

# パラメータ最適化の比較
python compare_parameters.py --csv test_data/corridor_grid.csv --save-dir comparison
```

詳細は `TOP_VIEW_TEST_README.md` を参照してください。

## トラブルシューティング

### よくある問題と解決方法

**カメラが認識されない**
```bash
# 利用可能なカメラデバイスの確認
ls /dev/video*
# 設定ファイルのdevice_idを適切な値に変更
```

**深度推定が遅い**
- GPU使用の確認（CUDAドライバーとTorchのGPU対応）
- モデルの軽量版への変更を検討
- グリッド圧縮の活用（grid.rows, grid.colsを調整）

**LAN接続エラー**
- IPアドレスとポート番号の確認
- ファイアウォールの設定確認
- ネットワーク接続の確認

**PointCloudの座標がおかしい**
- カメラ内部パラメータ（fx, fy, cx, cy）の調整
- scaling_factorの調整
- 圧縮データ使用時はcoord_mappingの確認

## パフォーマンス最適化

### 推奨設定

**リアルタイム性能重視**
```json
{
  "grid": {"rows": 6, "cols": 8},
  "depth": {"scaling_factor": 8.0},
  "camera": {"fps": 15}
}
```

**高精度重視**
```json
{
  "grid": {"rows": 12, "cols": 16},
  "depth": {"scaling_factor": 12.0},
  "camera": {"fps": 10}
}
```

## 技術仕様

システムの技術仕様とアルゴリズムについて：

### 設計文書

- **`設計.md`** - システム全体の設計思想、アーキテクチャ、API仕様
- **`障害物検知システムアルゴリズム.md`** - 深度変換から障害物検知までの詳細アルゴリズム仕様

### 可視化リソース

アルゴリズムの理解のために包括的な可視化を提供：

```bash
cd visualization
python coordinate_systems.py
```

生成される可視化：
- **座標系とアルゴリズム統合図** - 全変換プロセスの概観
- **個別詳細図** - カメラ座標系、深度変換、グリッド圧縮、トップダウン変換
- **パラメータ比較図** - 異なる設定での動作比較

詳細は `障害物検知システムアルゴリズム.md` の「8. 可視化リソース」セクションを参照してください。

### 座標系の概要

- **カメラ座標系**: X=右、Y=下、Z=前方向
- **画像座標系**: (u,v) ピクセル座標
- **グリッド座標系**: 圧縮された占有グリッド
- **ワールド座標系**: トップダウンビューでの実世界座標

## ライセンスと謝辞

このプロジェクトは以下のオープンソースプロジェクトを利用しています：

- **Depth Anything**: Meta AI Research 2024（Apache License 2.0）
- **OpenCV**: コンピュータビジョンライブラリ（Apache License 2.0）
- **FastAPI**: 高性能ウェブフレームワーク（MIT License）

本プロジェクトのコード（Depth Anythingモデル自体を除く）はMITライセンスの下で利用可能です。
