# Training
Modify ```config.yaml``` to set the model checkpoint, dataset paths and training parameters. A dataset with the COCO format is required. The dataset should consist of a folder with images and a JSON file with annotations. The JSON file should have the following structure:
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
sh train.sh
```

Note that training is done with nohup, so the script will continue running even if you close the terminal. The following flags can be passed to the training script:
- `--clear`: remove logs and checkpoints from previous runs when starting training.
- `--name`: name of the training run, used to create directories for logs and checkpoints.

## Logging

Logs and checkpoints are saved in the `logs` directory. You can monitor the training process using TensorBoard. To do this, run the following command in a separate terminal:
```bash
tensorboard --logdir=logs --port=6006
```
Then open your web browser and go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.
