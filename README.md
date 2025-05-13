# Huggingface Transformers

This repository demonstrates how to use the [Huggingface Transformers library](https://huggingface.co/docs/transformers/en/index) for object detection. Specifically it focuses on finetuning models on custom datasets. Currently, the following models have been tested but all transformers object detection models should work:

- [Conditional DETR](https://huggingface.co/docs/transformers/model_doc/conditional_detr)
- [D-FINE](https://huggingface.co/docs/transformers/main/en/model_doc/d_fine)
- [Deformable DETR](https://huggingface.co/docs/transformers/en/model_doc/deformable_detr)
- [RT-DETRv2](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)
- [YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos)

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
pip install transformers[torch] pycocotools scipy tensorboard albumentations
```

# Example dataset

The example dataset used in this repository is the [Traffic Signs Detection Dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection).

To download the dataset, install the kagglehub package and then run the following python script. This will download the dataset and extract it to a `data` folder, as well as create coco annotations for the dataset.
```bash
pip install kagglehub
python scripts/get_example_dataset.py
```

# Results

The following table summarizes the models´ performance on the test set of the example dataset. AP refers to mAP@50:5:95. The inference time (forward pass time) is measured on an NVIDIA T4 GPU with a batch size of 1.

| Checkpoint<br>(training epochs) |    AP   |   APs   |   APm   |   APl   | Inference Time (ms)<br> | Inference Time (ms)<br>(TensorRT) | Inference Time (ms)<br>(TensorRT, FP16*) | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Conditional DETR (50)|
|`conditional-detr-resnet-50`| 0.363 | 0.158 | 0.278 | 0.514 | 30.2 | | | 0.8 |
|D-FINE (50)|
|`dfine-small-obj365`|
|Deformable DETR (50)|
|`deformable-detr`| 0.699 | 0.337 | 0.553 | 0.849 | 38.8 | | | 16.5 |
|RT-DETRv2 (30)|
|`rtdetr_v2_r18vd`| 0.302 | 0.102 | 0.345 | 0.427 | 20.1 | 8.3 | 7.6 | 0.8 |
|`rtdetr_v2_r34vd`| 0.557 | 0.237 | 0.510 | 0.780 | 25.2 | 12.2 | 7.8 | 0.8 |
|`rtdetr_v2_r50vd`| 0.746 | 0.401 | 0.808 | 0.891 | 32.4 | 19.2 | 7.5 | 0.8 |
|`rtdetr_v2_r101vd`| 0.760 | 0.447 | 0.834 | 0.901 | 48.8 | 28.9 | 10.3 | 0.8 | 
|YOLOS (100)|
|`yolos-tiny`| 0.557 | 0.163 | 0.441 | 0.755 | 9.2 | | 5.7 | 0.8 |
|`yolos-small`| 0.579 | 0.150 | 0.520 | 0.823 | 10.4 | | 8.9 | 49.5 |
|`yolos-base`| 0.681 | 0.256 | 0.584 | 0.887 | 11.9 | | 9.9 | 132.2 |


*Note that using FP16 may lead to a slight difference in AP, in this case less than 0.01.
