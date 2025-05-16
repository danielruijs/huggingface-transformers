import os
import json
import torch
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor
import onnxruntime as ort
from dataclasses import dataclass


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def run_inference(image_processor, ort_session, image, threshold, use_fp16):
    inputs = image_processor(images=image, return_tensors="pt")

    pixel_values = inputs["pixel_values"]

    if use_fp16:
        pixel_values = pixel_values.numpy().astype(np.float16)
    else:
        pixel_values = pixel_values.numpy()

    with torch.no_grad():
        raw_outputs = ort_session.run(None, {"pixel_values": pixel_values})
        outputs = ModelOutput(
            logits=torch.from_numpy(raw_outputs[0]),
            pred_boxes=torch.from_numpy(raw_outputs[1]),
        )

    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=threshold,
    )
    return results[0]


def annotate_and_save(image, results, save_path, classes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(16)
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        text = f"{classes[int(label_id)]}: {score:.2f}"
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
        text_x = x1
        text_y = y1 - text_height - 2

        # Ensure text stays within image boundaries
        if text_y < 0:
            text_y = y1 + 2  # Place below the box if above goes out of bounds
        if text_x + text_width > image.width:
            text_x = image.width - text_width  # Adjust to fit within width

        draw.text((text_x, text_y), text, fill="red", font=font)

    image.save(save_path)
    print(f"Saved annotated image to {save_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model and image processor
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
    # Load the model configuration to get the classes
    with open(os.path.join(args.model_dir, "config.json"), "r") as f:
        model_config = json.load(f)
        classes = {int(k): v for k, v in model_config.get("id2label", {}).items()}

    image_paths = [
        os.path.join(args.img_dir, img_file)
        for img_file in os.listdir(args.img_dir)
        if img_file.endswith((".jpg", ".png"))
    ]

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        results = run_inference(
            image_processor=image_processor,
            ort_session=ort_session,
            image=image,
            threshold=args.threshold,
            use_fp16=args.fp16,
        )
        save_path = os.path.join(args.output_dir, os.path.basename(img_path))
        annotate_and_save(image, results, save_path, classes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory of the onnx model"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True, help="Directory for caching the engine"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory of the images to be processed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the images with predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
