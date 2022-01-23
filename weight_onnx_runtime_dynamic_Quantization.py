import torch.onnx 
from transformers import AutoTokenizer,AutoConfig,TensorType

import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType


import os

""" W.I.P   : We will bring about the simplest way to convert/export a transformere  to ONNX  soon."""
def Convert_ONNX(fin_model,input_size): 
    # set the model to inference mode 
    fin_model.eval() 
    """It's important to call model.eval() or model.train(False) before exporting the model, as this sets the model to inference mode. 
    This is needed since operators like dropout or batchnorm behave differently in inference and training mode."""

    # Let's create a dummy input tensor  
    #dummy_input = torch.randn(1, input_size, requires_grad=True)  

    """The need for ea dummy input is infact for the ONNX runtime to understand the possible input shape. This is not a dynamic value, thus
    its expected during the conversion .However,  dynamic_axes args such as these are there which do not seem to worl
    Refer:https://github.com/onnx/onnx/issues/654""" 


    """Use the command: 
    1.python -m transformers.convert_graph_to_onnx --framework pt --model /home/arjunkumbakkara/Desktop/onnxConversion/Best_Model/pytorch_model.bin 
    --quantize albert.onnx --opset 12
    """
    """
    2.python ../convert_graph_to_onnx.py --framework pt --model /home/arjunkumbakkara/Desktop/onnxConversion/Best_Model/pytorch_model.bin
     /home/arjunkumbakkara/Desktop/onnxConversion/Best_Model/romance.onnx"""

    """Loading Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("/home/arjunkumbakkara/Desktop/onnxConversion/Best_Model" , use_fast=True)
    dummy_input = dict(tokenizer(["find flights arriving new york city next saturday"], return_tensors="pt"))
     
    # Export the model   
    torch.onnx.export(fin_model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "flight.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=12,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 

    print('Model has been converted to ONNX')




""" For Clarity only consider the code from here to down below"""
"""Also, we have NOT removed the absolute paths of our local system in the code so that its understood that its absolute.Feel free to change it to configurable 
or other ways"""

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_opt_model,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

 
    print('ONNX full precision model size (MB):', os.path.getsize("/home/arjunkumbakkara/Desktop/onnx_compression/model_compressed.onnx")/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize("/home/arjunkumbakkara/Desktop/onnx_compression/model_quant.onnx")/(1024*1024))

if __name__ == "__main__": 
    quantize_onnx_model("/home/arjunkumbakkara/Desktop/onnx_compression/model_compressed.onnx","/home/arjunkumbakkara/Desktop/onnx_compression/model_quant.onnx")
    # Conversion to ONNX 
    #Convert_ONNX(f"/home/arjunkumbakkara/Desktop/onnxConversion/Best_Model/pytorch_model.bin",2)
