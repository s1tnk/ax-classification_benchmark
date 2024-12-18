import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader
from onnxruntime.quantization import QuantType
from PIL import Image
import os
from torchvision import transforms
from pathlib import Path
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder, input_name, batch_size=1):
        self.image_folder = image_folder
        self.input_name = input_name
        self.batch_size = batch_size
        self.image_list = []
        self.current_index = 0
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.image_list = [f for f in os.listdir(image_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Found {len(self.image_list)} images for calibration")

    def get_next(self):
        if self.current_index >= len(self.image_list):
            return None
            
        batch_images = []
        for _ in range(self.batch_size):
            if self.current_index >= len(self.image_list):
                break
                
            image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
            try:
                with Image.open(image_path).convert('RGB') as img:
                    processed_image = self.preprocess(img)
                    batch_images.append(processed_image.numpy())
                
            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")
                self.current_index += 1
                continue
                
            self.current_index += 1
        
        if not batch_images:
            return None
            
        batch_data = np.stack(batch_images)
        return {self.input_name: batch_data}

    def rewind(self):
        self.current_index = 0

def quantize_model_with_images(model_path, output_path, calibration_image_folder, batch_size=1):
    """ONNXモデルの量子化を実行"""
    try:
        model = onnx.load(model_path)
        input_name = model.graph.input[0].name
        
        calibration_reader = ImageCalibrationDataReader(
            calibration_image_folder,
            input_name,
            batch_size
        )
        
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantType.QUInt8
        )
        
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"Model: {Path(model_path).name}")
        logging.info(f"Original size: {original_size:.2f} MB")
        logging.info(f"Quantized size: {quantized_size:.2f} MB")
        logging.info(f"Compression ratio: {original_size/quantized_size:.2f}x")
        return True
        
    except Exception as e:
        logging.error(f"Error quantizing {model_path}: {e}")
        return False

def batch_quantize_models(input_folder, output_folder, calibration_folder):
    """フォルダ内のすべてのONNXモデルを量子化"""
    # 出力フォルダの作成
    os.makedirs(output_folder, exist_ok=True)
    
    # ONNXファイルの検索
    onnx_files = [f for f in os.listdir(input_folder) if f.endswith('.onnx')]
    
    if not onnx_files:
        logging.warning(f"No ONNX files found in {input_folder}")
        return
    
    logging.info(f"Found {len(onnx_files)} ONNX files to process")
    
    # 処理結果の集計用
    success_count = 0
    failed_count = 0
    
    # 各モデルの処理
    for onnx_file in onnx_files:
        input_path = os.path.join(input_folder, onnx_file)
        output_path = os.path.join(output_folder, f"{Path(onnx_file).stem}_static_quant_int8.onnx")
        
        logging.info(f"\nProcessing: {onnx_file}")
        
        if quantize_model_with_images(input_path, output_path, calibration_folder):
            success_count += 1
        else:
            failed_count += 1
    
    # 結果サマリーの出力
    logging.info("\n=== Quantization Summary ===")
    logging.info(f"Total models processed: {len(onnx_files)}")
    logging.info(f"Successfully quantized: {success_count}")
    logging.info(f"Failed: {failed_count}")

if __name__ == "__main__":
    input_folder = "./models"  # ONNXモデルが格納されているフォルダ
    output_folder = "./quantized_models"  # 量子化したモデルの出力先
    calibration_folder = "./calibration_images"  # キャリブレーション用画像フォルダ
    
    batch_quantize_models(input_folder, output_folder, calibration_folder)
