# ドキュメント索引

## プロジェクト概要

**障害物検知システム** - Depth Anythingモデルによるリアルタイム深度推定とLAN経由Point Cloud送信システム

## ドキュメント構成

### 1. 使用方法・セットアップ

- **[README.md](readme.md)** - システム概要、インストール、実行方法、設定例
  - 2つの実装方式（Linux Main API、Simple FastAPI Camera）
  - 設定ファイルの説明
  - トラブルシューティング

### 2. 技術仕様・設計

- **[設計.md](設計.md)** - システム全体の設計思想とアーキテクチャ
  - 設計思想とコンセプト
  - システム構成図
  - API仕様（Linux Main API）
  - 座標系定義

- **[障害物検知システムアルゴリズム.md](障害物検知システムアルゴリズム.md)** - 詳細アルゴリズム仕様
  - Depth Anything出力の深度変換アルゴリズム
  - 座標系変換の数学的定義
  - グリッド圧縮とマッピング
  - 占有グリッド生成
  - パラメータ調整ガイド
  - 可視化リソース

### 3. 可視化・検証ツール

- **[visualization/](visualization/)** - アルゴリズム理解のための可視化システム
  - `coordinate_systems.py` - 統合可視化生成スクリプト
  - `test_camera_view_fixed.py` - カメラ座標系単独テスト
  - `README.md` - 可視化システム使用ガイド
  
  **生成される可視化画像**:
  - **包括可視化**:
    - `comprehensive_visualization.png` - 4パネル統合アルゴリズム図
  - **個別詳細図**:
    - `camera_coordinate_system.png` - カメラ座標系とピンホールモデル
    - `depth_conversion_algorithm.png` - 深度変換アルゴリズム
    - `grid_compression_mapping.png` - グリッド圧縮マッピング
    - `topdown_transformation.png` - トップダウン変換
  - **カスタムパラメータ比較図**:
    - `custom_vis_s5.0_g4x3_r0.5_h0.2.png` - 低解像度設定
    - `custom_vis_s15.0_g16x12_r0.25_h0.15.png` - 高解像度設定

- **[tests/](tests/)** - テスト・検証スイート
  - `test_*.py` - 各コンポーネントの単体テスト
  - `generate_test_data.py` - テストデータ生成ツール
  - `conftest.py` - pytest設定ファイル
  
- **[test_data/](test_data/)** - テストデータセット
  - 各種シーン別のテスト用占有グリッドデータ（CSV形式）

### 4. ソースコード構成

```
kuma_depth_opt/
├── 実行ファイル
│   ├── linux_main.py           # Linux Main API（メイン処理）
│   ├── windows_client.py       # Windows クライアント
│   ├── simple_fastapi_camera.py # FastAPI カメラサーバー
│   └── utils.py               # 共通ユーティリティ
├── コア処理
│   └── depth_processor/
│       ├── __init__.py        # パッケージ初期化
│       ├── depth_model.py     # Depth Anythingモデル
│       ├── point_cloud.py     # PointCloud変換
│       └── visualization.py   # 可視化ユーティリティ
├── 設定ファイル
│   ├── config_linux.json     # Linux設定
│   ├── config_windows.json   # Windows設定
│   ├── config.json           # 汎用設定
│   └── requirements.txt      # Python依存関係
├── テスト・検証
│   ├── tests/                # 単体テストスイート
│   │   ├── conftest.py       # pytest設定
│   │   ├── test_*.py         # 各種単体テスト
│   │   └── generate_test_data.py # テストデータ生成
│   └── test_data/            # テストデータセット
│       └── *.csv             # 各種シーン別テストデータ
├── 可視化・検証ツール
│   └── visualization/
│       ├── README.md         # 可視化システム使用ガイド
│       ├── coordinate_systems.py # 可視化生成スクリプト
│       └── test_camera_view_fixed.py # カメラ座標系テスト
├── 生成された可視化画像
│   ├── comprehensive_visualization.png # 統合アルゴリズム図
│   ├── camera_coordinate_system.png    # カメラ座標系
│   ├── depth_conversion_algorithm.png  # 深度変換
│   ├── grid_compression_mapping.png    # グリッド圧縮
│   ├── topdown_transformation.png      # トップダウン変換
│   └── custom_vis_*.png               # カスタムパラメータ比較図
└── ドキュメント
    ├── readme.md              # システム概要・使用方法
    ├── 設計.md                # 設計仕様・アーキテクチャ
    ├── 障害物検知システムアルゴリズム.md # 詳細アルゴリズム仕様
    └── DOCUMENTATION_INDEX.md # ドキュメント索引（このファイル）
```

## ドキュメント利用ガイド

### 新規ユーザー向け

1. **[README.md](readme.md)** - システム概要と実行方法を理解
2. 設定ファイル例を参考に環境構築
3. **[visualization/](visualization/)** - アルゴリズムの動作を可視化で理解

### 開発者向け

1. **[設計.md](設計.md)** - システム全体アーキテクチャを把握
2. **[障害物検知システムアルゴリズム.md](障害物検知システムアルゴリズム.md)** - 詳細アルゴリズムを理解
3. ソースコード解析とテスト実行

### カスタマイズ・最適化

1. **[障害物検知システムアルゴリズム.md](障害物検知システムアルゴリズム.md)** の「7. パラメータ調整ガイド」
2. **[visualization/](visualization/)** でパラメータ変更の影響を可視化
3. 設定ファイルの調整

## バージョン情報

- **最終更新**: 2025年1月20日
- **プロジェクト整理**: ファイル構成整理、不要ファイル削除、テストファイル整理完了
- **ドキュメント構造化**: README分離、アルゴリズム仕様書作成、可視化統合完了

この索引により、プロジェクトの全体像を把握し、目的に応じた適切なドキュメントにアクセスできます。
