# Huggingface Transformers

This repository demonstrates how to use the [Huggingface Transformers library](https://huggingface.co/docs/transformers/en/index) for object detection. Specifically it focuses on finetuning models on custom datasets. Currently, the following models are supported:

- [RT-DETRv2](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)

See the respective folders for each model.

# Installation
Create a virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
# On Windows
source .venv/Scripts/activate
# On Linux/MacOS
source .venv/bin/activate
```

Install the required packages:
```bash
pip install transformers[torch] pycocotools scipy tensorboard
```

# Example dataset

The example dataset used in this repository is the [Traffic Signs Detection Dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection).

To download the dataset, install the kagglehub package and then run the following python script. This will download the dataset and extract it to a `data` folder, as well as create coco annotations for the dataset.
```bash
pip install kagglehub
python scripts/get_example_dataset.py
```