# ax-classification_benchmark


Environment
- MCU or Board name: "M5Stack Module-LLM(ax630c)"
- Operating System:
   "Linux m5stack-LLM 4.19.125 #1 SMP PREEMPT Thu Nov 14 17:40:17 CST 2024 aarch64 aarch64 aarch64 GNU/Linux"
- PyAXEngine version :"PyAXEngine 0.0.1 RC3"

## Model conversion
This runs on Ubuntu PC.
Download the model from Pytorch and perform static int8 quantization.


```
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
$ cd model_conv
$ mkdir -p models
$ mkdir -p quantized_models
$ mkdir -p calibration_images
$ cd models
$ python classification_torch_model_export.py
$ cd ..
$ python onnx_quantize_static_image_batch.py
```


## execution
This runs on M5Stack Module-LLM.

```
# root@LLM ./batch_run_onnx.sh
# root@LLM ./batch_run_axmodel.sh
```

![image](https://github.com/user-attachments/assets/63c68f97-feb2-4131-b449-8281e3a005f3)

