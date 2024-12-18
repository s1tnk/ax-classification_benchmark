# ax-classification_benchmark


| |CPU Process(onnxruntime)| |NPU Process(pyaxengine )| |
|:----|:----|:----|:----|:----|
| |defalult|onnx-quantized|ax-model(quant)|ax-model(quant)|
| |float32|int8|int8|int8|
| |--|--|NPU1(half core)|NPU2(full core)|
|shufflenet_v2_x0_5|38.2 |34.0 |1.1 |1.0shufflenet_v2_x1_0|91.3 |62.4 |1.4 |1.5shufflenet_v2_x1_5|154.9 |75.3 |1.7 |1.6shufflenet_v2_x2_0|276.8 |150.8 |5.5 |2.4mobilenet_v2|221.4 |88.5 |1.7 |1.4mobilenet_v3_large|206.2 |87.4 |2.4 |2.0squeezenet1_0|303.1 |241.2 |2.5 |1.5squeezenet1_1|173.0 |78.0 |2.8 |1.1resnet18|653.1 |321.1 |3.4 |3.1resnet34|1317.5 |645.8 |5.7 |5.1resnet50|1461.2 |687.1 |8.3 |6.1resnet152|4104.1 |1878.0 |17.1 |12.4vgg11|2529.1 |1427.9 |28.8 |26.2vgg13|3785.1 |1878.0 |31.9 |28.8vgg16|5141.4 |2620.2 |35.9 |32.0vgg19|6408.4 |3268.9 |39.9 |35.2|
