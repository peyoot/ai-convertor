# ai-convertor
Scripts to convert AI models 

## How to

Create a folder in tools and create environment:

```
mkdir ~/tools/
git clone https://github.com/peyoot/ai-convertor
cd ai-convertor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

#convert onnx model to TensorFlow
python3 onnx_to_tf.py onnx/mymodel.onnx saved_tf/

#convert Tensorflow to Tensorflow lite
tflite_convert --saved_model_dir=saved_tf --output_file=mymodel.tflite
```


 
