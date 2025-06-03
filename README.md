# Huggingface Transformers

This repository demonstrates how to use the [Huggingface Transformers library](https://huggingface.co/docs/transformers/en/index) for object detection. Specifically it focuses on finetuning models on custom datasets. The repository provides training, evaluation, and inference scripts which can be found in the [src](src) directory. The [src/onnx](src/onnx) directory contains instructions for exporting the models to the ONNX format and running evaluation and inference with TensorRT.

Currently, the following models have been tested but all transformers object detection models should work:

- [Conditional DETR](https://huggingface.co/docs/transformers/model_doc/conditional_detr)
- [D-FINE](https://huggingface.co/docs/transformers/main/en/model_doc/d_fine)
- [Deformable DETR](https://huggingface.co/docs/transformers/en/model_doc/deformable_detr)
- [DETR](https://huggingface.co/docs/transformers/model_doc/detr)
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
pip install transformers[torch] pycocotools scipy tensorboard albumentations tabulate
```

# Example datasets

Two example datasets are used in this repository:

## Dataset 1, 416x416 images

The first example dataset used in this repository is the [Traffic Signs Detection Dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection). The dataset contains 416x416 images of traffic signs.

To download the dataset, install the kagglehub package and then run the following python script. This will download the dataset and extract it to a `data` folder, as well as create coco annotations for the dataset.
```bash
pip install kagglehub
python scripts/get_example_dataset.py
```

## Dataset 2, 1920x1080 images

The second example dataset used in this repository is the [Brackish Underwater Dataset](https://public.roboflow.com/object-detection/brackish-underwater/1). The dataset contains 1920x1080 images of fish, crabs, and other marine animals. It can be downloaded by following the link and selecting the COCO download format.

# Results

The following tables summarizes the models´ performance on the test sets of the example datasets. AP refers to mAP@50:5:95. The inference time (forward pass time) is measured on an NVIDIA T4 GPU with a batch size of 1.

## Dataset 1, 416x416 images

| Checkpoint<br>(training epochs) |    AP   |   APs   |   APm   |   APl   | Inference Time (ms)<br> | Inference Time (ms)<br>AMP* | Inference Time (ms)<br>(TensorRT) | Inference Time (ms)<br>(TensorRT, FP16*) | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|RT-DETRv2 (30)|
|`rtdetr_v2_r18vd`| 0.801 | 0.553 | 0.831 | 0.908 | 20.1 | 27.4 | 8.3 | 7.6 | 0.8 |
|`rtdetr_v2_r34vd`| 0.809 | 0.576 | 0.845 | 0.911 | 25.2 | 34.2 | 12.2 | 7.8 | 0.8 |
|`rtdetr_v2_r50vd`| 0.798 | 0.498 | 0.834 | 0.903 | 32.4 | 41.9 | 19.2 | 7.5 | 0.8 |
|`rtdetr_v2_r101vd`|  |  |  |  | 48.8 | | 28.9 | 10.3 | 0.8 | 

*Note that using AMP or FP16 may lead to a slight difference in AP, in this case less than 0.01.

The following table summarizes the performance of RT-DETRv2, when trained with the [official implementation](https://github.com/lyuwenyu/RT-DETR) for 30 epochs.

| Variant<br> |    AP   |   APs   |   APm   |   APl   | Inference Time (ms) | Inference Time (ms)<br>AMP* | Inference Time (ms)<br>FP16* | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| RT-DETRv2-S<br>(rtdetr_v2_r18vd) | 0.790 | 0.564 | 0.780 | 0.899 | | 27.5 | | 0.5 |
| RT-DETRv2-M*<br>(rtdetr_v2_r34vd) | 0.813 | 0.611 | 0.775 | 0.914 | | 33.2 | | 0.5 |
| RT-DETRv2-M<br>(rtdetr_v2_r50vd) | 0.808 | 0.592 | 0.791 | 0.915 | | 35.5 | | 0.5 |
| RT-DETRv2-L<br>(rtdetr_v2_r50vd) | 0.817 | 0.627 | 0.785 | 0.910 | | 41.9 | | 0.5 |

## Dataset 2, 1920x1080 images

| Checkpoint<br>(training epochs) |    AP   |   APs   |   APm   |   APl   | Inference Time (ms) | Inference Time (ms)<br>AMP* | Inference Time (ms)<br>FP16* | Inference Time (ms)<br>(TensorRT) | Inference Time (ms)<br>(TensorRT, FP16*) | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|RT-DETRv2 (30)|
|`rtdetr_v2_r18vd`| 0.801 | 0.220 | 0.787 | 0.824 | 87.8 | 57.7 | 43.1 | 65.9 | 31.3 | 1.1 |

*Note that using AMP or FP16 may lead to a slight difference in AP, in this case less than 0.01.

The following table summarizes the performance of RT-DETRv2, when trained with the [official implementation](https://github.com/lyuwenyu/RT-DETR) for 20 epochs.

| Variant<br> |    AP   |   APs   |   APm   |   APl   | Inference Time (ms) | Inference Time (ms)<br>AMP* | Inference Time (ms)<br>FP16* | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| RT-DETRv2-S<br>(rtdetr_v2_r18vd) | 0.803 | 0.196 | 0.786 | 0.846 | | 56.6 | | 0.9 |
