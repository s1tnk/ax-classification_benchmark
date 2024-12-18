# Copyright (c) 2024 nnn112358


import onnxruntime
import numpy as np
from PIL import Image
import argparse
import time

def load_model(model_path):
    """ONNXモデルを読み込む"""
    try:
        session = onnxruntime.InferenceSession(model_path)
        return session
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return None

def preprocess_image(image_path, target_size=(256, 256), crop_size=(224, 224)):
    """画像の前処理を行う"""
    try:
        # 画像を読み込み
        img = Image.open(image_path).convert("RGB")
        
        # オリジナルサイズを取得
        original_width, original_height = img.size
        
        # 短い辺を基準に中央クロップ
        if original_width < original_height:
            crop_area = original_width
        else:
            crop_area = original_height
            
        crop_x = (original_width - crop_area) // 2
        crop_y = (original_height - crop_area) // 2
        
        # 中央正方形にクロップ
        img = img.crop((crop_x, crop_y, crop_x + crop_area, crop_y + crop_area))
        
        # 指定サイズにリサイズ
        img = img.resize(target_size)
        
        # 中央をクロップ
        crop_x = (target_size[0] - crop_size[0]) // 2
        crop_y = (target_size[1] - crop_size[1]) // 2
        img = img.crop((crop_x, crop_y, crop_x + crop_size[0], crop_y + crop_size[1]))
        
        # numpy配列に変換
        img_array = np.array(img).astype(np.float32)

        # 値を0-1の範囲に正規化
        img_array = img_array / 255.0

        # ImageNetの平均値と標準偏差で正規化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        img_array = (img_array - mean) / std

        # チャネルの順序を(H, W, C)から(C, H, W)に変更
        img_array = np.transpose(img_array, (2, 0, 1))

        img_array = np.expand_dims(img_array, axis=0)  # バッチ次元を追加
        
        return img_array
    except Exception as e:
        print(f"画像の前処理に失敗しました: {e}")
        return None

def get_top_k_predictions(output, k=5):
    """Top-K予測を取得"""
    try:
        top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
        top_k_scores = output[0].flatten()[top_k_indices]
        return top_k_indices, top_k_scores
    except Exception as e:
        print(f"予測の取得に失敗しました: {e}")
        return None, None

def main(model_path, image_path, target_size, crop_size, k):
    """メイン処理"""
    # モデルを読み込む
    session = load_model(model_path)
    if session is None:
        return
    
    preprocess_start = time.perf_counter()  # 前処理開始時間
    # 画像を前処理
    input_tensor = preprocess_image(image_path, target_size, crop_size)
    if input_tensor is None:
        return

    preprocess_time = (time.perf_counter() - preprocess_start) * 1000  # ミリ秒に変換
    print(f"前処理時間: {preprocess_time:.2f} ms")


# 推論を5回実行して平均時間を計算する
    num_trials = 10
    total_inference_time = 0
    all_times = []  # 全ての推論時間を保存するリスト

    for i in range(num_trials):
    # 推論開始時間を記録
        inference_start = time.perf_counter()
    
    # 入力名を取得して推論を実行
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor})
    
    # 推論時間を計測
        inference_time = (time.perf_counter() - inference_start) * 1000  # ミリ秒に変換
        total_inference_time += inference_time
        all_times.append(inference_time)

        print(f"Trial {i + 1}: 推論時間: {inference_time:.2f} ms")

    # 平均推論時間を計算:trial 2から9までの平均を計算
    selected_times = all_times[1:9]  # インデックス1（2番目）から8（9番目）までを選択
    average_inference_time = sum(selected_times) / len(selected_times)
    print(f"\n2-9番目の試行の平均推論時間: {average_inference_time:.2f} ms")


    # Top-K予測を取得
    top_k_indices, top_k_scores = get_top_k_predictions(output, k)
    if top_k_indices is None:
        return
    
    # 結果を表示
    print(f"Top {k} Predictions:")
    for i in range(k):
        print(f"Class Index: {top_k_indices[i]}, Score: {top_k_scores[i]:.4f}")

if __name__ == "__main__":

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='ONNXモデルを使用した画像分類')
    parser.add_argument('--model', type=str, default="models/torch_vision_mobilenet_v2.onnx",
                      help='ONNXモデルのパス')
    parser.add_argument('--image', type=str, default="cat.jpg",
                      help='入力画像のパス')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                      help='リサイズサイズ (幅 高さ)')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[224, 224],
                      help='クロップサイズ (幅 高さ)')
    parser.add_argument('--topk', type=int, default=5,
                      help='表示する予測数')

    args = parser.parse_args()
    
    # パラメータ設定
    MODEL_PATH = args.model
    IMAGE_PATH = args.image
    TARGET_SIZE = tuple(args.target_size)
    CROP_SIZE = tuple(args.crop_size)
    K = args.topk
    
    print(f"モデルパス: {MODEL_PATH}")
    print(f"画像パス: {IMAGE_PATH}")
    print(f"リサイズサイズ: {TARGET_SIZE}")
    print(f"クロップサイズ: {CROP_SIZE}")
    print(f"Top-K: {K}\n")
    
    # メイン処理を実行
    main(MODEL_PATH, IMAGE_PATH, TARGET_SIZE, CROP_SIZE, K)
    
    
    
