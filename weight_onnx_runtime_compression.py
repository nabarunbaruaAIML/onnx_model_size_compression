
"""@Authors: Arjun Kumbakkara , Nabarun Barua"""



import onnx
from onnxruntime.transformers.onnx_model import OnnxModel

def has_same_value(val_one,val_two):
  if val_one.raw_data == val_two.raw_data:
    return True
  else:
    return False


""" Excerpt from ONNX Team on the Correctness of the solution: 
ALBERT model has shared weights among layers as part of the optimization from BERT . 
The export  torch.onnx.export outputs the weights to different tensors as so model size becomes larger.
Using the below python Script we can remove duplication of weights, and reduce model size
ie,  Compare each pair of initializers, when they are the same, just remove one initializer, and update all reference of it to the other initializer."""
"""ONNX Team @tianleiwu"""


"""Case: AlBERT model trained for text classification clocked at 46.8mb of size of the .bin weights file. When converted to the ONNX runtime , it became 345mb . 
We tried all optimizations on python before the conversion. However, the way out was to convert the .onnx converted weights to a compressed version""" 



if __name__ == "__main__":
  path = f"/home/arjunkumbakkara/Desktop/onnx_compression/model.onnx"
  model=onnx.load(path)
  onnx_model=OnnxModel(model)
  output_path = f"model_compressed.onnx"
  count = len(model.graph.initializer)
  same = [-1] * count
  for i in range(count - 1):
    if same[i] >= 0:
      continue
    for j in range(i+1, count):
      if has_same_value(model.graph.initializer[i], model.graph.initializer[j]):
        same[j] = i

  for i in range(count):
    if same[i] >= 0:
      onnx_model.replace_input_of_all_nodes(model.graph.initializer[i].name, model.graph.initializer[same[i]].name)

onnx_model.update_graph()
onnx_model.save_model_to_file(output_path)