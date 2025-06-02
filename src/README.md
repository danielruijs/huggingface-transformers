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
- Disabling denoising by setting `disable_denoising: True` in the config file yielded significantly better results on the two example datasets.

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
python evaluation.py --model_dir path/to/model/checkpoint --cocoann_file path/to/coco/annotations.json --img_dir path/to/images --threshold 0.01 --precision fp32 --lowmem
```
- `--image_dir`: May be omitted if the annotation file contains full paths to the images.
- `--threshold`: Confidence threshold for predictions. Default is 0.01.
- `--precision`: Sets the precision for evaluation. Options are `fp32`, `fp16` or `amp`. Default is `fp32`. Note that using `fp16` or `amp` may lead to slightly lower AP (less than 0.01 with the example datasets).
- `--lowmem`: Reduces memory usage during evaluation. This may slow down the process but is useful if memory is limited.

# Inference
To run inference, use the following command:
```bash
python inference.py --model_dir path/to/model --img_dir path/to/images --output_dir path/to/output --threshold 0.5 --fp16
```
- `--image_dir`: Directory of the images to be processed.
- `--output_dir`: Directory where the images with predictions will be saved.
- `--threshold`: Confidence threshold for predictions. Default is 0.5.
- `--fp16`: Runs the model with FP16 precision.
