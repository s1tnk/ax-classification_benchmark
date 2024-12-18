import os
from onnxsim import simplify

# 入力フォルダと出力フォルダの設定
input_folder = "quantized_models"
output_folder = "quantized_models_sim"

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# 入力フォルダ内のファイルを走査
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".onnx"):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, file)
            
            try:
                print(f"Processing: {input_path}")
                
                # ONNXモデルの簡略化
                model_simplified, check = simplify(input_path)
                
                # 簡略化が成功したか確認
                if not check:
                    print(f"Failed to simplify: {input_path}")
                    continue
                
                # 結果を保存
                with open(output_path, "wb") as f:
                    f.write(model_simplified.SerializeToString())
                
                print(f"Simplified model saved to: {output_path}")
            
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
