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
|D-FINE (50)|
|`dfine-small-obj365`|
|Deformable DETR (50)|
|`deformable-detr`| 0.699 | 0.337 | 0.553 | 0.849 | 38.8 | | | | 16.5 |
|RT-DETRv2 (30)|
|`rtdetr_v2_r18vd`| 0.302 | 0.102 | 0.345 | 0.427 | 20.1 | 27.4 | 8.3 | 7.6 | 0.8 |
|`rtdetr_v2_r34vd`| 0.557 | 0.237 | 0.510 | 0.780 | 25.2 | | 12.2 | 7.8 | 0.8 |
|`rtdetr_v2_r50vd`| 0.746 | 0.401 | 0.808 | 0.891 | 32.4 | | 19.2 | 7.5 | 0.8 |
|`rtdetr_v2_r101vd`| 0.760 | 0.447 | 0.834 | 0.901 | 48.8 | | 28.9 | 10.3 | 0.8 | 
|YOLOS (100)|
|`yolos-tiny`| 0.557 | 0.163 | 0.441 | 0.755 | 9.2 | | | 5.7 | 0.8 |
|`yolos-small`| 0.579 | 0.150 | 0.520 | 0.823 | 10.4 | | | 8.9 | 49.5 |
|`yolos-base`| 0.681 | 0.256 | 0.584 | 0.887 | 11.9 | | | 9.9 | 132.2 |

*Note that using AMP or FP16 may lead to a slight difference in AP, in this case less than 0.01.

The following table summarizes the performance of RT-DETRv2, when trained with the [official implementation](https://github.com/lyuwenyu/RT-DETR) for 30 epochs.

| Variant<br> |    AP   |   APs   |   APm   |   APl   | Inference Time (ms) | Inference Time (ms)<br>AMP* | Inference Time (ms)<br>FP16* | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| RT-DETRv2-S<br>(rtdetr_v2_r18vd) | 0.790 | 0.564 | 0.780 | 0.899 | | 27.5 | | 0.5 |

## Dataset 2, 1920x1080 images

| Checkpoint<br>(training epochs) |    AP   |   APs   |   APm   |   APl   | Inference Time (ms) | Inference Time (ms)<br>AMP* | Inference Time (ms)<br>FP16* | Inference Time (ms)<br>(TensorRT) | Inference Time (ms)<br>(TensorRT, FP16*) | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|RT-DETRv2 (20)|
|`rtdetr_v2_r18vd`| 0.592 | 0.065 | 0.544 | 0.651 | 87.8 | 57.7 | 43.1 | 65.9 | 31.3 | 1.1 |

*Note that using AMP or FP16 may lead to a slight difference in AP, in this case less than 0.01.

The following table summarizes the performance of RT-DETRv2, when trained with the [official implementation](https://github.com/lyuwenyu/RT-DETR) for 20 epochs.

| Variant<br> |    AP   |   APs   |   APm   |   APl   | Inference Time (ms) | Inference Time (ms)<br>AMP* | Inference Time (ms)<br>FP16* | Post-processing time (ms)<br> |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| RT-DETRv2-S<br>(rtdetr_v2_r18vd) | 0.803 | 0.196 | 0.786 | 0.846 | | 56.6 | | 0.9 |
