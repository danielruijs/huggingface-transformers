import torch
import time
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor
import onnxruntime as ort
from torch.utils.data import DataLoader
from dataclasses import dataclass
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from coco_utils import COCODataset, compute_COCO_metrics, create_size_map
from train import serialize_tensor_dict

EVAL_RESULTS_JSONL = "eval_results.jsonl"


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def run_inference(
    image_processor, ort_session, dataset, threshold, size_map, use_fp16, use_lowmem
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    labels = []
    total_forward = 0
    total_post = 0
    warmup_batches = 5  # Number of batches to not include in timing
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = batch["pixel_values"]
        batch_labels = batch["labels"]

        # Forward pass
        start_forward = time.perf_counter()
        with torch.no_grad():
            if use_fp16:
                inputs = inputs.numpy().astype(np.float16)
            else:
                inputs = inputs.numpy()
            raw_outputs = ort_session.run(None, {"pixel_values": inputs})
            outputs = ModelOutput(
                logits=torch.from_numpy(raw_outputs[0]),
                pred_boxes=torch.from_numpy(raw_outputs[1]),
            )
        end_forward = time.perf_counter()

        # Post-processing of outputs
        start_post = time.perf_counter()
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=[size_map[int(batch_labels["image_id"])]],
            threshold=threshold,
        )
        end_post = time.perf_counter()

        if i >= warmup_batches:
            total_forward += end_forward - start_forward
            total_post += end_post - start_post

        if use_lowmem:
            with open(EVAL_RESULTS_JSONL, "a") as f:
                record = {
                    "predictions": serialize_tensor_dict(results[0]),
                    "labels": serialize_tensor_dict(batch_labels),
                }
                f.write(json.dumps(record) + "\n")
        else:
            predictions.extend(results)
            labels.append(batch_labels)

    avg_forward_time = total_forward / (len(dataloader) - warmup_batches)
    avg_post_time = total_post / (len(dataloader) - warmup_batches)
    print(
        f"\nTotal batches processed: {len(dataloader)} (of which {warmup_batches} were not included in timing)"
    )
    print(f"Average forward pass time: {avg_forward_time * 1000:.4f} ms")
    print(f"Average post-processing time: {avg_post_time * 1000:.4f} ms\n")

    if use_lowmem:
        with open(EVAL_RESULTS_JSONL, "r") as f:
            for line in f:
                rec = json.loads(line)
                predictions.append(rec["predictions"])
                labels.append(rec["labels"])
    return predictions, labels


def main(args):
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    ort_session = ort.InferenceSession(
        args.model_dir + "/model.onnx",
        providers=[
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": args.cache_dir,
                },
            )
        ],
    )

    dataset = COCODataset(
        cocoann_file=args.cocoann_file,
        image_processor=image_processor,
        image_root=args.img_dir,
    )

    predictions, labels = run_inference(
        image_processor=image_processor,
        ort_session=ort_session,
        dataset=dataset,
        threshold=args.threshold,
        size_map=create_size_map(cocoann_file=args.cocoann_file),
        use_fp16=args.fp16,
        use_lowmem=args.lowmem,
    )

    _ = compute_COCO_metrics(
        predictions=predictions, labels=labels, cocoann_file=args.cocoann_file
    )

    if os.path.exists(EVAL_RESULTS_JSONL):
        os.remove(EVAL_RESULTS_JSONL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory of the onnx model"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True, help="Directory for caching the engine"
    )
    parser.add_argument(
        "--cocoann_file",
        type=str,
        required=True,
        help="Path to COCO annotations file",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="",
        help="Image directory. May be omitted if the annotation file contains full paths to the images.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Confidence threshold for predictions",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--lowmem", action="store_true", help="Use low memory mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
