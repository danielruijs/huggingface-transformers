# ONNX and TensorRT

To export the model to ONNX and to run inference with TensorRT, you need to install the following packages:

Install the latest version of `optimum` from the Hugging Face GitHub repository and the `onnx` and `onnxruntime-gpu` packages:
```bash
python -m pip install git+https://github.com/huggingface/optimum.git
pip install onnx onnxruntime-gpu
```

Then install TensorRT by following the instructions in the [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).
The following commands can be used on Debian 11 (Bullseye). Make sure to download the correct version of TensorRT for your system which can be found at the [NVIDIA developer website](https://developer.nvidia.com/tensorrt/download).
```bash
mkdir ~/tensorrt && cd ~/tensorrt

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.10.0/tars/TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz

tar -xvzf TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz

sudo mv TensorRT-10.10.0.31/lib/* /usr/lib/x86_64-linux-gnu/

sudo ldconfig

rm -rf ~/tensorrt
```

## Exporting to ONNX

To export the model to the ONNX format, specify the model checkpoint directory and the ouput directory where the ONNX model will be saved and run the following command:
```bash
optimum-cli export onnx --model path/to/model/checkpoint --task object-detection --opset 17 --device cuda --dtype fp16 output/directory
```
The `--dtype fp16` flag can be omitted if you want to export the model in FP32.

## Evaluation

To run evaluation on the exported ONNX model with TensorRT, use the following command:
```bash
python onnx/evaluation.py --model_dir path/to/onnx/model --cache_dir path/to/cache --cocoann_file path/to/coco/annotations.json --img_dir path/to/images --threshold 0.01 --fp16 --lowmem
```
- `--cache_dir`: Directory where the model engine is cached. The first time the model is run, the engine will be created and saved in this directory. Note that this can take some time.
- `--image_dir`: May be omitted if the annotation file contains full paths to the images.
- `--threshold`: Sets the confidence threshold for predictions. Default is 0.01.
- `--fp16`: Runs the model with FP16 precision. Must be set to the same value as the one used during export. Note that this may lead to slightly lower AP (less than 0.01 with the example datasets).
- `--lowmem`: Reduces memory usage during evaluation. This may slow down the process but is useful if memory is limited.

## Inference

To run inference with TensorRT, run the following command:
```bash
python onnx/inference.py --model_dir path/to/onnx/model --cache_dir path/to/cache --img_dir path/to/images --output_dir path/to/output --threshold 0.5 --fp16
```
- `--image_dir`: Directory of the images to be processed.
- `--output_dir`: Directory where the images with predictions will be saved.
- `--threshold`: Confidence threshold for predictions. Default is 0.5.
- `--fp16`: Runs the model with FP16 precision. Must be set to the same value as the one used during export.