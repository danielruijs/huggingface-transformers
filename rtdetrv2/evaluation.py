import torch
import argparse
import time
from tqdm import tqdm
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from coco_utils import COCODataset
from torch.utils.data import DataLoader
from coco_utils import compute_COCO_metrics


def run_inference(
    image_processor,
    model,
    dataset,
    threshold,
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()

    predictions = []
    labels = []
    total_forward = 0
    total_post = 0
    warmup_batches = 5  # Number of batches to not include in timing
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = batch["pixel_values"].to(model.device)
        batch_labels = batch["labels"]

        # Forward pass
        start_forward = time.perf_counter()
        with torch.no_grad():
            outputs = model(inputs)
        end_forward = time.perf_counter()

        # Post-processing of outputs
        start_post = time.perf_counter()
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=batch_labels["size"],
            threshold=threshold,
        )
        end_post = time.perf_counter()

        if i >= warmup_batches:
            total_forward += end_forward - start_forward
            total_post += end_post - start_post

        predictions.extend(results)
        labels.append(batch_labels)

    avg_forward_time = total_forward / (len(dataloader) - warmup_batches)
    avg_post_time = total_post / (len(dataloader) - warmup_batches)
    print(
        f"\nTotal batches processed: {len(dataloader)} (of which {warmup_batches} were not included in timing)"
    )
    print(f"Average forward pass time: {avg_forward_time * 1000:.4f} ms")
    print(f"Average post-processing time: {avg_post_time * 1000:.4f} ms\n")
    return predictions, labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_processor = RTDetrImageProcessor.from_pretrained(args.model_dir)
    model = RTDetrV2ForObjectDetection.from_pretrained(args.model_dir).to(device)

    dataset = COCODataset(
        cocoann_file=args.cocoann_file,
        image_processor=image_processor,
        image_root=args.img_dir,
    )

    predictions, labels = run_inference(
        image_processor=image_processor,
        model=model,
        dataset=dataset,
        threshold=args.threshold,
    )

    _ = compute_COCO_metrics(
        predictions=predictions, labels=labels, cocoann_file=args.cocoann_file
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory of the model checkpoint"
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
