import cv2
import time
import numpy as np
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from collections import deque
from contextlib import asynccontextmanager

# Depth Anything用
from depth_processor import DepthProcessor, create_depth_visualization, create_depth_grid_visualization
# ★ create_default_depth_image をインポート
from depth_processor.visualization import create_default_depth_image
# 天頂視点マップ用の関数をインポート
from depth_processor import convert_to_absolute_depth, depth_to_point_cloud, create_top_down_occupancy_grid, visualize_occupancy_grid

# FastAPIのライフサイクル管理を最新のasynccontextmanagerに変更
@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時の処理
    print("アプリケーション起動: カメラとスレッドを初期化します")
    # スレッドは既にグローバルで開始されているので、ここでは何もしない
    
    yield  # アプリケーション実行中
    
    # 終了時の処理
    print("アプリケーション終了: リソースを解放します")
    try:
        # カメラのクリーンアップ
        if cap is not None:
            cap.release()
            print("カメラリソースを解放しました")
    except Exception as e:
        print(f"終了処理中のエラー: {e}")

# FastAPIアプリケーションをlifespanコンテキストマネージャーで初期化
app = FastAPI(lifespan=lifespan)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# カメラバッファ設定とエラー処理を追加
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小に
cap.set(cv2.CAP_PROP_FPS, 30)        # カメラのFPS設定

# カメラの接続状態を確認
if not cap.isOpened():
    print("エラー: カメラに接続できません")
    import sys
    sys.exit(1)
else:
    print(f"カメラ接続成功: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

depth_processor = DepthProcessor()

# 共有メモリを拡張
latest_depth_map = None
latest_camera_frame = None
latest_depth_grid = None # ★追加: 圧縮済み深度グリッド用
frame_timestamp = 0
depth_map_lock = threading.Lock() # latest_depth_map, latest_camera_frame, latest_depth_grid の保護用
last_inference_time = 0
INFERENCE_INTERVAL = 0.08
GRID_COMPRESSION_SIZE = (12, 16) # グリッド圧縮サイズ (rows, cols)

# カメラ内部パラメータ (仮の値 - 実際にはキャリブレーションで取得)
# raw_depth_map (256x384) に対応する値を想定
ORIGINAL_DEPTH_HEIGHT = 256
ORIGINAL_DEPTH_WIDTH = 384
FX = 332.5  # 例: 384 / (2 * tan(60deg_hfov / 2))
FY = 309.0  # 例: 256 / (2 * tan(45deg_vfov / 2))
CX = ORIGINAL_DEPTH_WIDTH / 2.0
CY = ORIGINAL_DEPTH_HEIGHT / 2.0

# カメラキャプチャ専用スレッド（修正）
def camera_capture_thread():
    global latest_camera_frame, frame_timestamp
    consecutive_errors = 0
    max_errors = 5
    
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                with depth_map_lock:
                    latest_camera_frame = frame.copy()
                    frame_timestamp = time.time()
                consecutive_errors = 0  # エラーカウンタをリセット
            else:
                consecutive_errors += 1
                print(f"カメラ読み取りエラー ({consecutive_errors}/{max_errors})")
                
                if consecutive_errors >= max_errors:
                    print("カメラをリセットします...")
                    cap.release()
                    time.sleep(1.0)
                    cap.open(0)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_errors = 0
        except Exception as e:
            print(f"カメラ例外: {e}")
            time.sleep(0.5)
            
        time.sleep(0.05)  # 20FPSを維持

# 処理時間計測用
camera_times = deque(maxlen=1000)
inference_times = deque(maxlen=1000)
compression_times = deque(maxlen=1000) # ★追加: 圧縮時間用
visualization_times = deque(maxlen=1000)
encoding_times = deque(maxlen=1000)

# パフォーマンス統計用の変数を追加
fps_stats = {
    "camera": deque(maxlen=30),
    "depth": deque(maxlen=30),
    "grid": deque(maxlen=30),
    "inference": deque(maxlen=30),
    "compression": deque(maxlen=30), # ★追加: 圧縮FPS用
    "top_down": deque(maxlen=30)
}
last_frame_times = {
    "camera": 0,
    "depth": 0,
    "grid": 0,
}

def log_processing_times():
    """5秒ごとに平均、最大、最小の処理時間をログ出力"""
    while True:
        time.sleep(5)
        if camera_times:
            print(f"[Camera] Avg: {np.mean(camera_times):.4f}s, Max: {np.max(camera_times):.4f}s, Min: {np.min(camera_times):.4f}s")
        if inference_times:
            print(f"[Inference] Avg: {np.mean(inference_times):.4f}s, Max: {np.max(inference_times):.4f}s, Min: {np.min(inference_times):.4f}s")
        if compression_times: # ★追加
            print(f"[Compression] Avg: {np.mean(compression_times):.4f}s, Max: {np.max(compression_times):.4f}s, Min: {np.min(compression_times):.4f}s")
        if visualization_times:
            print(f"[Visualization] Avg: {np.mean(visualization_times):.4f}s, Max: {np.max(visualization_times):.4f}s, Min: {np.min(visualization_times):.4f}s")
        if encoding_times:
            print(f"[Encoding] Avg: {np.mean(encoding_times):.4f}s, Max: {np.max(encoding_times):.4f}s, Min: {np.min(encoding_times):.4f}s")

threading.Thread(target=log_processing_times, daemon=True).start()

# 推論専用の関数を追加
def inference_thread():
    global latest_depth_map, last_inference_time, latest_depth_grid # ★ latest_depth_grid をグローバル変数として追加
    
    # 設定値確認用デバッグ出力
    print(f"[Thread] GRID_COMPRESSION_SIZE: {GRID_COMPRESSION_SIZE}")
    print(f"[Thread] Camera Parameters: FX={FX}, FY={FY}, CX={CX}, CY={CY}")
    
    while True:
        current_time = time.time()
        if current_time - last_inference_time > INFERENCE_INTERVAL:
            with depth_map_lock:
                if latest_camera_frame is None:
                    time.sleep(0.01)
                    continue
                frame_for_inference = latest_camera_frame.copy()
                current_capture_time = frame_timestamp
            
            # 推論用にリサイズ
            small_frame = cv2.resize(frame_for_inference, (128, 128), interpolation=cv2.INTER_AREA)
            
            # 深度推定
            start_time_inference = time.perf_counter()
            raw_depth_map, _ = depth_processor.predict(small_frame)
            inference_duration = time.perf_counter() - start_time_inference
            inference_times.append(inference_duration)
            
            # 深度マップ圧縮
            compressed_grid = None
            compression_duration = 0
            if raw_depth_map is not None:
                start_time_compression = time.perf_counter()
                compressed_grid = depth_processor.compress_depth_to_grid(raw_depth_map, grid_size=GRID_COMPRESSION_SIZE)
                compression_duration = time.perf_counter() - start_time_compression
                compression_times.append(compression_duration)
                if compressed_grid is None:
                    print("[Thread] compress_depth_to_grid returned None.") 
            else:
                print("[Thread] depth_processor.predict returned None, skipping compression.")

            processing_end_time = time.time()
            
            with depth_map_lock:
                latest_depth_map = raw_depth_map # 元の深度マップも保持（他の用途があるかもしれないため）
                latest_depth_grid = compressed_grid # ★圧縮グリッドを保存
                last_inference_time = processing_end_time # ★処理完了時刻を更新
            
            delay = processing_end_time - current_capture_time
            print(f"[Thread] Inference: {inference_duration:.4f}s, Compression: {compression_duration:.4f}s, Total: {(inference_duration + compression_duration):.4f}s, Delay: {delay*1000:.1f}ms")
        else:
            time.sleep(0.01) # 推論間隔までの待機

# カメラスレッド起動
threading.Thread(target=camera_capture_thread, daemon=True).start()

# 推論スレッド起動
threading.Thread(target=inference_thread, daemon=True).start()

def get_depth_stream():
    while True:
        # 共有メモリからカメラフレームと深度マップを取得
        with depth_map_lock:
            if latest_depth_map is None or latest_camera_frame is None:
                time.sleep(0.01)
                continue
            current_depth_map = latest_depth_map.copy()
            current_frame = latest_camera_frame.copy()  # 現在のフレームも取得

        # 深度マップの可視化
        start_time = time.perf_counter()
        vis = create_depth_visualization(current_depth_map, (128, 128))
        vis = cv2.resize(vis, (320, 240), interpolation=cv2.INTER_NEAREST)
        visualization_times.append(time.perf_counter() - start_time)

        # FPS計算とテキスト表示
        now = time.time()
        if last_frame_times["depth"] > 0:
            fps = 1.0 / (now - last_frame_times["depth"])
            fps_stats["depth"].append(fps)
        last_frame_times["depth"] = now

        # FPS表示と遅延表示
        if len(fps_stats["depth"]) > 0:
            avg_fps = sum(fps_stats["depth"]) / len(fps_stats["depth"])
            cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 最新の遅延情報を表示
            with depth_map_lock:
                delay = (time.time() - frame_timestamp) * 1000
            cv2.putText(vis, f"Delay: {delay:.1f}ms", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # JPEG エンコード
        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.015)  # 約66FPSに向上（0.02→0.015に変更）

# カメラストリームを提供する関数を追加
def get_camera_stream():
    """カメラ映像のストリームを提供する関数"""
    print("[CameraStream] Camera stream started")
    
    # キープアライブフレームを作成（データがない場合の表示用）
    fallback_image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(fallback_image, "Waiting for camera...", (60, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ret, fallback_buffer = cv2.imencode('.jpg', fallback_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    
    error_count = 0
    
    while True:
        try:
            current_frame = None
            current_timestamp = 0
            
            # 共有メモリからカメラフレームを取得
            with depth_map_lock:
                if latest_camera_frame is None:
                    error_count += 1
                    if error_count > 5:  # 5回連続でフレームがない場合はフォールバックイメージを表示
                        print("[CameraStream] No camera frame available, using fallback")
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fallback_buffer.tobytes() + b'\r\n')
                        error_count = 0  # リセット
                    time.sleep(0.1)
                    continue
                current_frame = latest_camera_frame.copy()
                current_timestamp = frame_timestamp
                error_count = 0  # フレーム取得に成功したのでリセット
            
            if current_frame is None:
                time.sleep(0.1)
                continue
                
            # カメラ処理時間測定開始
            start_time = time.perf_counter()
            
            # FPSの計算
            now = time.time()
            if last_frame_times["camera"] > 0:
                fps = 1.0 / (now - last_frame_times["camera"])
                fps_stats["camera"].append(fps)
            last_frame_times["camera"] = now
            
            # 情報をオーバーレイ表示
            if len(fps_stats["camera"]) > 0:
                avg_fps = sum(fps_stats["camera"]) / len(fps_stats["camera"])
                cv2.putText(current_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
                # 遅延表示
                delay = (time.time() - current_timestamp) * 1000
                cv2.putText(current_frame, f"Delay: {delay:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 処理時間計測
            camera_times.append(time.perf_counter() - start_time)
            
            # JPEG エンコード
            start_time = time.perf_counter()
            ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            encoding_times.append(time.perf_counter() - start_time)
            if not ret:
                continue
                
            # フレームを送信
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.015)  # 約66FPS
            
        except Exception as e:
            print(f"[CameraStream] Error: {e}")
            import traceback
            print(traceback.format_exc())
            
            # エラー時にはフォールバックを表示
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fallback_buffer.tobytes() + b'\r\n')
            time.sleep(0.1)

def get_depth_grid_stream():
    while True:
        current_compressed_grid_to_visualize = None
        with depth_map_lock:
            if latest_depth_grid is not None:
                current_compressed_grid_to_visualize = latest_depth_grid.copy()
            # latest_camera_frame はここでは不要

        if current_compressed_grid_to_visualize is None:
            time.sleep(0.01) # グリッドデータがまだない場合は待機
            continue

        start_time_vis = time.perf_counter()
        # ★ create_depth_grid_visualization に圧縮済みグリッドとcell_sizeのみを渡す
        grid_img = create_depth_grid_visualization(current_compressed_grid_to_visualize, cell_size=20) 
        
        if grid_img is None or len(grid_img.shape) < 2:
            grid_img = create_default_depth_image(width=320, height=240) # フォールバック
        elif len(grid_img.shape) == 2 or (len(grid_img.shape) == 3 and grid_img.shape[2] == 1):
            grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        
        grid_img = cv2.resize(grid_img, (320, 240), interpolation=cv2.INTER_NEAREST)
        visualization_times.append(time.perf_counter() - start_time_vis)

        # FPS計算とテキスト表示
        now = time.time()
        if last_frame_times["grid"] > 0:
            fps = 1.0 / (now - last_frame_times["grid"])
            fps_stats["grid"].append(fps)
        last_frame_times["grid"] = now

        # FPS表示と遅延表示
        if len(fps_stats["grid"]) > 0:
            avg_fps = sum(fps_stats["grid"]) / len(fps_stats["grid"])
            cv2.putText(grid_img, f"FPS: {avg_fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 最新の遅延情報を表示
            with depth_map_lock:
                delay = (time.time() - frame_timestamp) * 1000
            cv2.putText(grid_img, f"Delay: {delay:.1f}ms", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # JPEG エンコード
        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', grid_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.015)  # 約66FPSに向上（0.02→0.015に変更）

@app.get("/stats")
async def get_stats():
    """統計情報を取得するAPIエンドポイント"""
    # 共有メモリからフレームタイムスタンプを取得して遅延を計算
    with depth_map_lock:
        current_delay = (time.time() - frame_timestamp) * 1000 if frame_timestamp > 0 else 0
        
    # 中央値を計算するヘルパー関数
    def median(values):
        if not values:
            return 0
        values_list = list(values)
        values_list.sort()
        return values_list[len(values_list) // 2]
        
    stats = {
        "fps": {
            # 平均値の代わりに中央値を使用
            "camera": round(median(fps_stats["camera"]), 1) if fps_stats["camera"] else 0,
            "depth": round(median(fps_stats["depth"]), 1) if fps_stats["depth"] else 0,
            "grid": round(median(fps_stats["grid"]), 1) if fps_stats["grid"] else 0,
            "inference": round(median(fps_stats["inference"]), 1) if fps_stats["inference"] else 0,
        },
        "latency": {
            "camera": round(np.mean(camera_times) * 1000, 1) if camera_times else 0,
            "inference": round(np.mean(inference_times) * 1000, 1) if inference_times else 0,
            "visualization": round(np.mean(visualization_times) * 1000, 1) if visualization_times else 0,
            "encoding": round(np.mean(encoding_times) * 1000, 1) if encoding_times else 0,
            "total_delay": round(current_delay, 1)  # 現在の総遅延を追加
        }
    }
    return stats

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Fast Camera Streaming</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { display: flex; flex-wrap: wrap; gap: 15px; }
            .video-box { background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); position: relative; min-height: 240px; width: 320px; }
            .video-box img { display: block; width: 100%; height: auto; }
            .video-box .loading-indicator { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #666; }
            h2 { margin-top: 0; color: #333; }
            .stats { margin-top: 20px; padding: 10px; background: #e8f5e9; border-radius: 5px; }
            #stats-container { font-family: monospace; }
            .retry-button { margin-top: 5px; background: #2196F3; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Fast Depth Processing System</h1>
        
        <div class="container">
            <div class="video-box" id="camera-box">
                <h2>Camera Stream</h2>
                <div class="loading-indicator">Loading camera stream...</div>
                <img src="/video" alt="Camera Stream" onload="this.parentNode.querySelector('.loading-indicator').style.display='none';" onerror="this.style.display='none'; this.parentNode.querySelector('.loading-indicator').innerHTML='Failed to load camera stream <button class=\'retry-button\' onclick=\'retryStream(this, \\\"/video\\\")\'>Retry</button>';" />
            </div>
            <div class="video-box" id="depth-box">
                <h2>Depth Map</h2>
                <div class="loading-indicator">Loading depth map...</div>
                <img src="/depth_video" alt="Depth Map" onload="this.parentNode.querySelector('.loading-indicator').style.display='none';" onerror="this.style.display='none'; this.parentNode.querySelector('.loading-indicator').innerHTML='Failed to load depth map <button class=\'retry-button\' onclick=\'retryStream(this, \\\"/depth_video\\\")\'>Retry</button>';" />
            </div>
            <div class="video-box" id="grid-box">
                <h2>Depth Grid</h2>
                <div class="loading-indicator">Loading depth grid...</div>
                <img src="/depth_grid" alt="Depth Grid" onload="this.parentNode.querySelector('.loading-indicator').style.display='none';" onerror="this.style.display='none'; this.parentNode.querySelector('.loading-indicator').innerHTML='Failed to load depth grid <button class=\'retry-button\' onclick=\'retryStream(this, \\\"/depth_grid\\\")\'>Retry</button>';" />
            </div>
            <div class="video-box" id="topdown-box">
                <h2>Top-Down View</h2>
                <div class="loading-indicator">Loading top-down view...</div>
                <img src="/top_down_view" alt="Top-Down View" onload="this.parentNode.querySelector('.loading-indicator').style.display='none';" onerror="this.style.display='none'; this.parentNode.querySelector('.loading-indicator').innerHTML='Failed to load top-down view <button class=\'retry-button\' onclick=\'retryStream(this, \\\"/top_down_view\\\")\'>Retry</button>';" />
            </div>
        </div>
        <div class="stats">
            <h3>Performance Stats</h3>
            <div id="stats-container">Loading stats...</div>
        </div>
        
        <script>
            // ストリームの再試行関数
            function retryStream(button, streamUrl) {
                const box = button.closest('.video-box');
                const loadingIndicator = box.querySelector('.loading-indicator');
                loadingIndicator.innerHTML = 'Reconnecting...';
                
                const img = box.querySelector('img') || document.createElement('img');
                img.style.display = 'block';
                img.src = streamUrl + '?retry=' + new Date().getTime(); // キャッシュ回避のためのクエリパラメータ
                img.onload = function() {
                    loadingIndicator.style.display = 'none';
                };
                img.onerror = function() {
                    img.style.display = 'none';
                    loadingIndicator.innerHTML = 'Failed to load stream <button class="retry-button" onclick="retryStream(this, \'' + streamUrl + '\')">Retry</button>';
                };
                
                if (!box.contains(img)) {
                    box.appendChild(img);
                }
            }

            // 2秒ごとに統計情報を更新
            setInterval(async () => {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    const container = document.getElementById('stats-container');
                    
                    let html = '<table>';
                    html += '<tr><th>Stream</th><th>FPS</th><th>Latency (ms)</th></tr>';
                    html += `<tr><td>Camera</td><td>${stats.fps.camera}</td><td>${stats.latency.camera}</td></tr>`;
                    html += `<tr><td>Depth</td><td>${stats.fps.depth}</td><td>-</td></tr>`;
                    html += `<tr><td>Grid</td><td>${stats.fps.grid}</td><td>-</td></tr>`;
                    html += `<tr><td>Inference</td><td>${stats.fps.inference}</td><td>${stats.latency.inference}</td></tr>`;
                    html += `<tr><td>Visualization</td><td>-</td><td>${stats.latency.visualization}</td></tr>`;
                    html += `<tr><td>Total Delay</td><td>-</td><td>${stats.latency.total_delay}</td></tr>`;
                    html += '</table>';
                    
                    container.innerHTML = html;
                } catch (e) {
                    console.error('Failed to fetch stats:', e);
                }
            }, 2000);
        </script>
    </body>
    </html>
    """

@app.get("/video")
async def video():
    return StreamingResponse(get_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video():
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_grid")
async def depth_grid():
    return StreamingResponse(get_depth_grid_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/top_down_view")
async def top_down_view():
    print("[API] Top-Down View endpoint called")
    return StreamingResponse(get_top_down_view_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

def get_top_down_view_stream():
    """天頂視点マップのストリームを提供する関数"""
    # パフォーマンス測定用
    last_frame_time = 0
    fps_stats["top_down"] = deque(maxlen=30)
    first_frame = True
    
    print("[TopDownStream] Top-down view stream started")
    
    # キープアライブフレームを初期化(何もデータがない場合でも表示するため)
    no_data_image = create_default_depth_image(width=320, height=240, text="Top-Down View: Waiting for data...")
    no_data_buffer = None
    try:
        ret, no_data_buffer = cv2.imencode('.jpg', no_data_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    except Exception as e:
        print(f"[TopDownStream] Warning: Failed to create keep-alive frame: {e}")
    
    # 設定パラメータ
    scaling_factor = 15.0  # 深度スケーリング係数
    grid_resolution = 0.1  # グリッドの解像度（メートル/セル）
    grid_width = 100       # グリッドの幅（セル数）
    grid_height = 100      # グリッドの高さ（セル数）
    height_threshold = 0.3 # 通行可能と判定する高さの閾値（メートル）
    
    # グローバル変数の状態確認
    print("[TopDownStream] グローバル変数状態チェック:")
    print(f"[TopDownStream] GRID_COMPRESSION_SIZE: {GRID_COMPRESSION_SIZE}")
    print(f"[TopDownStream] Camera Parameters: FX={FX}, FY={FY}, CX={CX}, CY={CY}")
    print(f"[TopDownStream] ORIGINAL_DEPTH_HEIGHT: {ORIGINAL_DEPTH_HEIGHT}, ORIGINAL_DEPTH_WIDTH: {ORIGINAL_DEPTH_WIDTH}")
    
    # エラーカウンター
    error_count = 0
    last_error_time = 0
    
    # キープアライブフレームを最初に送信
    if no_data_buffer is not None:
        print("[TopDownStream] Sending initial keep-alive frame")
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + no_data_buffer.tobytes() + b'\r\n')
    
    # メインループ
    while True:
        current_grid_data = None
        current_raw_depth_map = None

        try:
            with depth_map_lock:
                if latest_depth_grid is not None:
                    current_grid_data = latest_depth_grid.copy()
                if latest_depth_map is not None:
                    current_raw_depth_map = latest_depth_map.copy()
            
            if current_grid_data is None or current_raw_depth_map is None:
                # データがない場合は待機して「No Data」画像を送信
                current_time = time.time()
                if current_time - last_error_time > 5:  # 5秒ごとにログ出力
                    print("[TopDownStream] Waiting for depth data...")
                    last_error_time = current_time
                
                # データがない状態が続いたらキープアライブフレームを送信
                error_count += 1
                if error_count > 5 and no_data_buffer is not None:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + no_data_buffer.tobytes() + b'\r\n')
                    error_count = 0  # カウンターリセット
                
                time.sleep(0.1)  # 待機時間
                continue
            
            # ここからデータ処理開始
            start_time_vis = time.perf_counter()
            
            try:
                # 圧縮グリッドデータを絶対深度に変換
                original_height, original_width = current_raw_depth_map.shape[:2]
                print(f"[TopDownStream] Processing depth data: grid={current_grid_data.shape}, raw={current_raw_depth_map.shape}")
                
                # 絶対深度へ変換
                print(f"[TopDownStream] Converting grid to absolute depth with scaling_factor={scaling_factor}")
                absolute_depth_grid = convert_to_absolute_depth(current_grid_data, scaling_factor=scaling_factor)
                if absolute_depth_grid is not None:
                    print(f"[TopDownStream] Absolute depth range: {np.min(absolute_depth_grid):.2f}m to {np.max(absolute_depth_grid):.2f}m")
                
                # 点群生成
                print(f"[TopDownStream] Generating point cloud with camera parameters: fx={FX}, fy={FY}, cx={CX}, cy={CY}")
                point_cloud = depth_to_point_cloud(
                    absolute_depth_grid,
                    fx=FX, fy=FY, cx=CX, cy=CY,
                    is_grid_data=True,
                    original_height=original_height,
                    original_width=original_width,
                    grid_rows=GRID_COMPRESSION_SIZE[0],
                    grid_cols=GRID_COMPRESSION_SIZE[1]
                )
                
                # 点群から占有グリッド生成
                if point_cloud is None or point_cloud.size == 0:
                    print("[TopDownStream] No valid points in point cloud")
                    vis_img = create_default_depth_image(width=320, height=240, text="No Valid Point Cloud")
                else:
                    point_count = point_cloud.shape[0]
                    print(f"[TopDownStream] Generated point cloud with {point_count} points")
                    print(f"[TopDownStream] Point cloud data type: {point_cloud.dtype}, shape: {point_cloud.shape}")
                    
                    # 点群データの統計情報を追加
                    if point_count > 0:
                        # XYZ座標の基本統計
                        x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
                        y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
                        z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
                        print(f"[TopDownStream] Point cloud range - X: {x_min:.2f} to {x_max:.2f}m, Y: {y_min:.2f} to {y_max:.2f}m, Z: {z_min:.2f} to {z_max:.2f}m")
                        
                        # 値の分布を確認
                        x_mean, y_mean, z_mean = np.mean(point_cloud[:, 0]), np.mean(point_cloud[:, 1]), np.mean(point_cloud[:, 2])
                        print(f"[TopDownStream] Point cloud means - X: {x_mean:.2f}m, Y: {y_mean:.2f}m, Z: {z_mean:.2f}m")
                        
                        # Y値（高さ）の分布を詳しく確認（床検出に重要）
                        y_percentiles = np.percentile(point_cloud[:, 1], [5, 25, 50, 75, 95])
                        print(f"[TopDownStream] Height (Y) percentiles [5,25,50,75,95]: {y_percentiles}")
                    
                    # 占有グリッド生成
                    print(f"[TopDownStream] Creating occupancy grid: resolution={grid_resolution}m, size={grid_width}x{grid_height}, height_threshold={height_threshold}m")
                    occupancy_grid = create_top_down_occupancy_grid(
                        point_cloud, 
                        grid_resolution=grid_resolution,
                        grid_width=grid_width,
                        grid_height=grid_height,
                        height_threshold=height_threshold
                    )
                    
                    # 占有グリッドの視覚化
                    if occupancy_grid is not None:
                        print(f"[TopDownStream] Occupancy grid created with shape: {occupancy_grid.shape}")
                        unique_values = np.unique(occupancy_grid)
                        print(f"[TopDownStream] Grid values: {unique_values}")
                        
                        vis_img = visualize_occupancy_grid(occupancy_grid, scale_factor=3)
                        print(f"[TopDownStream] Occupancy grid visualized: {vis_img.shape}")
                    else:
                        print("[TopDownStream] Failed to create occupancy grid")
                        vis_img = create_default_depth_image(width=320, height=240, text="Invalid Grid")
                
                # リサイズとフォーマット調整
                vis_img = cv2.resize(vis_img, (320, 240), interpolation=cv2.INTER_NEAREST)
                if len(vis_img.shape) == 2:
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
                
                # 成功したのでエラーカウンターをリセット
                error_count = 0
                
            except Exception as e:
                print(f"[TopDownStream] Error in processing: {e}")
                import traceback
                print(traceback.format_exc())
                vis_img = create_default_depth_image(width=320, height=240, text=f"Error: {str(e)[:20]}")
            
            # FPS計算
            visualization_times.append(time.perf_counter() - start_time_vis)
            now = time.time()
            if last_frame_times.get("top_down", 0) > 0:
                fps = 1.0 / (now - last_frame_times["top_down"])
                fps_stats["top_down"].append(fps)
            last_frame_times["top_down"] = now
            
            # テキストオーバーレイ
            if len(fps_stats["top_down"]) > 0:
                avg_fps = sum(fps_stats["top_down"]) / len(fps_stats["top_down"])
                cv2.putText(vis_img, f"FPS: {avg_fps:.1f}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                with depth_map_lock:
                    delay = (time.time() - frame_timestamp) * 1000 if frame_timestamp > 0 else 0
                cv2.putText(vis_img, f"Delay: {delay:.1f}ms", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # JPEG エンコード
            ret, buffer = cv2.imencode('.jpg', vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                print("[TopDownStream] Failed to encode image")
                continue
            
            # フレーム送信
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as e:
            print(f"[TopDownStream] Unhandled error: {e}")
            import traceback
            print(traceback.format_exc())
            
            # エラー時にもキープアライブフレームを送信
            if no_data_buffer is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + no_data_buffer.tobytes() + b'\r\n')
        
        # フレームレート制御
        time.sleep(0.1)  # 10 FPS 程度に制限（Top-Down処理は重いため）
# 例外ハンドリングを強化
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_msg = f"予期せぬエラーが発生しました: {str(exc)}"
    print(f"[エラー] {error_msg}")
    import traceback
    print(traceback.format_exc())
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg}
    )

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8888)
    except KeyboardInterrupt:
        print("Ctrl+Cが押されました。アプリケーションを終了します。")
    except Exception as e:
        print(f"予期せぬエラーでアプリケーションが終了しました: {e}")
        import traceback
        print(traceback.format_exc())