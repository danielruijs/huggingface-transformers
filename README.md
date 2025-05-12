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

# Results

The following table summarizes the models´ performance on the test set of the example dataset. AP refers to mAP@50:5:95. The inference time (forward pass time) in milliseconds is measured on an NVIDIA T4 GPU with a batch size of 1. All models are trained for 30 epochs.

### RT-DETRv2
| Checkpoint |    AP   |   APs   |   APm   |   APl   | Inference Time | Inference Time<br>(TensorRT) | Inference Time<br>(TensorRT, FP16*) |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|`rtdetr_v2_r18vd`| 0.302 | 0.102 | 0.345 | 0.427 | 20.1 | 8.3 | 7.6 |
|`rtdetr_v2_r34vd`| 0.557 | 0.237 | 0.510 | 0.780 | 25.2 | 12.2 | 7.8 |
|`rtdetr_v2_r50vd`| 0.746 | 0.401 | 0.808 | 0.891 | 32.4 | 19.2 | 7.5 |

*Note that using FP16 may lead to a slight difference in AP, in this case less than 0.01.