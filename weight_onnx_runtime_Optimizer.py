from pyexpat import model
import torch.onnx 
from transformers import AutoTokenizer,AutoConfig,TensorType
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import os


# optimize transformer-based models with onnxruntime-tools
from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions



#W.I.P


def optimizeUsingONNX_tools(model):
    opt_options = BertOptimizationOptions('bert')
    opt_options.enable_embed_layer_norm = False
    opt_model = optimizer.optimize_model(
        model,
        'bert',
        num_heads=12,
        hidden_size=768,
        optimization_options=opt_options)
    opt_model.save_model_to_file(,"/home/arjunkumbakkara/Desktop/onnx_compression/model_quant_optimizer.onnx")

if __name__ == "__main__": 
    optimizeUsingONNX_tools("/home/arjunkumbakkara/Desktop/onnx_compression/model_compressed.onnx")
    