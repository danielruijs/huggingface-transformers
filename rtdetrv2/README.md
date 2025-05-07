# Installation
```bash
pip install transformers torch pycocotools pyyaml
```

# Training
Modify ```config.yaml``` to set the model checkpoint, dataset paths and training parameters. A dataset with the COCO format is required. The dataset should consist of a folder with images and a JSON file with annotations. The JSON file should have the following structure:
```json
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

Then run the training script:
```bash
sh train.sh
```