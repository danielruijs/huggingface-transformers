# Training
Select the config file for the model you want to train from the `configs` directory and specify the model checkpoint, dataset paths and training parameters. A dataset with the COCO format is required. The dataset should consist of a folder with images and a JSON file with annotations. The JSON file should have the following structure:
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

Make sure to move to the `src` directory before running the training script:
```bash
cd src
```

Then run the training script:
```bash
sh train.sh --config path/to/train/config.yaml
```

Note that training is done with nohup, so training will continue even if you close the terminal. The output from the training script will be saved in the `train.log` file. The following flags can be passed to the training script:
- `--clear`: remove all logs and checkpoints from previous runs when starting training.
- `--name`: name of the training run, used to create directories for logs and checkpoints.

The best model from training is saved in the `output_dir` directory specified in the config file. The image processor is saved along with the best model to be used for inference.

## Tips for Training

- If you get OOM errors during evaluation, try setting `batch_eval_metrics: True` in the config file.

## Logging

Logs and checkpoints are saved in the `logs` directory. You can monitor the training process using TensorBoard. To do this, run the following command in a separate terminal:
```bash
tensorboard --logdir logs/ --port 6006
```
Then open your web browser and go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

## Scheduling

To schedule multiple training runs, you can use the `schedule_training.sh` script. It will run the training script for each config file in `config_directory`.
```bash
rm -f train.log && nohup sh schedule_training.sh config_directory >> train.log 2>&1 &
```

# Evaluation
To evaluate the model, run the evaluation script:
```bash
python evaluation.py --model_dir path/to/model/checkpoint --cocoann_file path/to/coco/annotations.json --img_dir path/to/images --threshold 0.01 --fp16 --lowmem
```
- `--image_dir`: May be omitted if the annotation file contains full paths to the images.
- `--threshold`: Confidence threshold for predictions. Default is 0.01.
- `--fp16`: Runs the model in FP16 mode. Note that this may lead to slightly lower AP (less than 0.01 with the example datasets).
- `--lowmem`: Reduces memory usage during evaluation. This may slow down the process but is useful if memory is limited.

# Inference
To run inference, use the following command:
```bash
python inference.py --model_dir path/to/model --img_dir path/to/images --output_dir path/to/output --threshold 0.5 --fp16
```
- `--image_dir`: Directory of the images to be processed.
- `--output_dir`: Directory where the images with predictions will be saved.
- `--threshold`: Confidence threshold for predictions. Default is 0.5.
- `--fp16`: Runs the model in FP16 mode.

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
- `--fp16`: Runs the model in FP16 mode. Must be set to the same value as the one used during export. Note that this may lead to slightly lower AP (less than 0.01 with the example datasets).
- `--lowmem`: Reduces memory usage during evaluation. This may slow down the process but is useful if memory is limited.

## Inference

To run inference with TensorRT, run the following command:
```bash
python onnx/inference.py --model_dir path/to/onnx/model --cache_dir path/to/cache --img_dir path/to/images --output_dir path/to/output --threshold 0.5 --fp16
```
- `--image_dir`: Directory of the images to be processed.
- `--output_dir`: Directory where the images with predictions will be saved.
- `--threshold`: Confidence threshold for predictions. Default is 0.5.
- `--fp16`: Runs the model in FP16 mode. Must be set to the same value as the one used during export.