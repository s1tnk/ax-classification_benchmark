# ax-classification_benchmark


Environment
- MCU or Board name: "M5Stack Module-LLM(ax630c)"
- Operating System:
   "Linux m5stack-LLM 4.19.125 #1 SMP PREEMPT Thu Nov 14 17:40:17 CST 2024 aarch64 aarch64 aarch64 GNU/Linux"
- PyAXEngine version :"PyAXEngine 0.0.1 RC3"

## Model conversion
This runs on Ubuntu PC.
Download the model from Pytorch and perform static int8 quantization.
After this,Convert with Pulsar2. See sample mobilenetv2.

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
root@m5stack-LLM:# wget https://github.com/AXERA-TECH/pyaxengine/releases/download/0.0.1rc3/axengine-0.0.1-py3-none-any.whl
root@m5stack-LLM:# pip install axengine-0.0.1-py3-none-any.whl
root@m5stack-LLM:# ./batch_run_onnx.sh
root@m5stack-LLM:# ./batch_run_axmodel.sh
```


## Result 
This is Process Time[msec]

| Model Name | CPU Process(onnxruntime) | CPU Process(onnxruntime) | NPU Process(pyaxengine) | NPU Process(pyaxengine) |
|------------|-------------------------|-------------------------|----------------------|----------------------|
|            | default (float32) | onnx-quantized (int8) | ax-model(quant) NPU1(half core) (int8) | ax-model(quant) NPU2(full core) (int8) |
| shufflenet_v2_x0_5 | 38.2 | 34.0 | 1.1 | 1.0 |
| shufflenet_v2_x1_0 | 91.3 | 62.4 | 1.4 | 1.5 |
| shufflenet_v2_x1_5 | 154.9 | 75.3 | 1.7 | 1.6 |
| shufflenet_v2_x2_0 | 276.8 | 150.8 | 5.5 | 2.4 |
| mobilenet_v2 | 221.4 | 88.5 | 1.7 | 1.4 |
| mobilenet_v3_large | 206.2 | 87.4 | 2.4 | 2.0 |
| squeezenet1_0 | 303.1 | 241.2 | 2.5 | 1.5 |
| squeezenet1_1 | 173.0 | 78.0 | 2.8 | 1.1 |
| resnet18 | 653.1 | 321.1 | 3.4 | 3.1 |
| resnet34 | 1317.5 | 645.8 | 5.7 | 5.1 |
| resnet50 | 1461.2 | 687.1 | 8.3 | 6.1 |
| resnet152 | 4104.1 | 1878.0 | 17.1 | 12.4 |
| vgg11 | 2529.1 | 1427.9 | 28.8 | 26.2 |
| vgg13 | 3785.1 | 1878.0 | 31.9 | 28.8 |
| vgg16 | 5141.4 | 2620.2 | 35.9 | 32.0 |
| vgg19 | 6408.4 | 3268.9 | 39.9 | 35.2 |

## Note:

Since the first execution of PyAXEngine was slow, I measured the execution time of the 2nd to 9th executions. See below.<br>
https://github.com/AXERA-TECH/pyaxengine/issues/17  
