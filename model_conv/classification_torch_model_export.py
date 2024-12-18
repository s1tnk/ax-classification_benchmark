import os
import torch
import torchvision

def pytorch_model_to_onnx(model_type, model_name):
    """
    Downloads a PyTorch model and converts it to ONNX format
    
    Args:
        model_type (str): The model repository (e.g., 'pytorch/vision:v0.14.0')
        model_name (str): The name of the model to download
    """
    print(f"Processing {model_name}...")
    
    # Download and create the model
    model = torch.hub.load(model_type, model_name, pretrained=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_file = f'./torch_vision_{model_name}.onnx'
    torch.onnx.export(
        model,                  # model being run
        dummy_input,           # model input (or a tuple for multiple inputs)
        onnx_file,            # where to save the model
        export_params=True,    # store the trained parameter weights inside the model file
        opset_version=11,      # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],   # the model's input names
        output_names=['output'], # the model's output names
#        dynamic_axes={
#            'input': {0: 'batch_size'},    # variable length axes
#            'output': {0: 'batch_size'}
#        }
    )
    

if __name__ == "__main__":    
    # Print available models
    print("Available models:")
    print(torch.hub.list('pytorch/vision:v0.14.0'))
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    os.chdir('models')
    
    # List of models to convert
    models = [
        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
        'resnet18', 'resnet34', 'resnet50', 'resnet152',
        'squeezenet1_0', 'squeezenet1_1',
        'vgg11', 'vgg13', 'vgg16', 'vgg19'
    ]
    
    # Convert each model
    model_type = 'pytorch/vision:v0.14.0'
    for model_name in models:
        try:
            pytorch_model_to_onnx(model_type, model_name)
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")

