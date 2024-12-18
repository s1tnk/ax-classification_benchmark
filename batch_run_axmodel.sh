#!/bin/bash

# モデルが格納されているフォルダ
ONNX_FOLDER="axmodels_npu2"
# 推論に使用する画像
IMAGE_PATH="cat.jpg"
# 実行するPythonスクリプト
SCRIPT_PATH="classification_run_ax.py"
# ログファイル名
LOG_FILE="ax_execution_log.txt"

# フォルダ内のONNXファイルを検索
ONNX_FILES=$(ls $ONNX_FOLDER/*.axmodel 2>/dev/null)

# ログファイルを初期化
echo "実行ログ: $(date)" > "$LOG_FILE"

# ONNXファイルが存在しない場合のエラーメッセージ
if [ -z "$ONNX_FILES" ]; then
  echo "フォルダ '$ONNX_FOLDER' 内にAXファイルが見つかりませんでした。" | tee -a "$LOG_FILE"
  exit 1
fi

# 各ONNXファイルを処理
for MODEL_PATH in $ONNX_FILES; do
  echo "\nモデル '$(basename $MODEL_PATH)' を実行中..." | tee -a "$LOG_FILE"
  python3 "$SCRIPT_PATH" --model "$MODEL_PATH" --image "$IMAGE_PATH" >> "$LOG_FILE" 2>&1
  
  # 実行結果の確認
  if [ $? -ne 0 ]; then
    echo "モデル '$(basename $MODEL_PATH)' の実行中にエラーが発生しました。" | tee -a "$LOG_FILE"
  fi
done
