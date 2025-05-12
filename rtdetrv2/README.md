# Training
Modify `configs/train.yaml` to set the model checkpoint, dataset paths and training parameters. A dataset with the COCO format is required. The dataset should consist of a folder with images and a JSON file with annotations. The JSON file should have the following structure:
```
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080,
    }
    // More image entries...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 500,
      "iscrowd": 0
    }
    // More annotation entries...
  ],
  "categories": [
    {
      "id": 0,
      "name": "category_name",
    }
    // More category entries...
  ]
}
```

Make sure to move to the `rtdetrv2` directory before running the training script:
```bash
cd rtdetrv2
```

Then run the training script:
```bash
sh train.sh --config path/to/train/config.yaml
```

Note that training is done with nohup, so training will continue even if you close the terminal. The following flags can be passed to the training script:
- `--clear`: remove all logs and checkpoints from previous runs when starting training.
- `--name`: name of the training run, used to create directories for logs and checkpoints.

The best model from training is saved in the `output_dir` directory specified in the config file. The image processor is saved in the same directory along with the model to be used for inference.

## Logging

Logs and checkpoints are saved in the `logs` directory. You can monitor the training process using TensorBoard. To do this, run the following command in a separate terminal:
```bash
tensorboard --logdir=logs --port=6006
```
Then open your web browser and go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

# Inference
To run inference, modify the `configs/inference.yaml` file to set the model checkpoint and the paths to the folder with the images you want to run inference on. Then run the inference script:
```bash
python inference.py --config path/to/inference/config.yaml
```

# Evaluation
To evaluate the model, modify the `configs/evaluation.yaml` file to set the model checkpoint and the path to the COCO format dataset to evaluate on. Then run the evaluation script:
```bash
python evaluation.py --config path/to/eval/config.yaml
```

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

To export the model to the ONNX format, specify the model checkpoint directory and the ouput directory where the ONNX model will be saved and run the following command:
```bash
optimum-cli export onnx --model path/to/model/checkpoint --task object-detection --opset 17 --device cuda --dtype fp16 output/directory
```

## Evaluation

To run evaluation on the exported ONNX model with TensorRT, use the following command:
```bash
python onnx/evaluation.py --model_dir path/to/onnx/model --cache_dir path/to/cache --cocoann_file path/to/coco/annotations.json --image_dir path/to/images --threshold 0.01
```
The `--cache_dir` parameter is the directory where the model engine is cached. The first time the model is run, the engine will be created and saved in this directory. Note that this can take some time. The `--image_dir` parameter may be omitted if the annotation file contains full paths to the images. The `--threshold` parameter is the confidence threshold for the predictions. The default value is 0.01.

## Inference

To run inference with TensorRT, run the following command:
```bash
python onnx/inference.py --model_dir path/to/onnx/model --cache_dir path/to/cache --img_dir path/to/images --output_dir path/to/output --threshold 0.01
```
The `--image_dir` parameter is the directory of the images to be processed. The `--output_dir` parameter is the directory where the output images will be saved. The `--threshold` parameter is the confidence threshold for the predictions with a default value of 0.5.
