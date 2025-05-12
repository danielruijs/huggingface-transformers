import torch
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection


def run_inference(image_processor, model, image, threshold):
    inputs = image_processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    image_processor = RTDetrImageProcessor.from_pretrained(args.model_dir)
    model = RTDetrV2ForObjectDetection.from_pretrained(args.model_dir).to(device)
    classes = model.config.id2label

    image_paths = [
        os.path.join(args.img_dir, img_file)
        for img_file in os.listdir(args.img_dir)
        if img_file.endswith((".jpg", ".png"))
    ]

    model.eval()
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        results = run_inference(image_processor, model, image, args.threshold)
        save_path = os.path.join(args.output_dir, os.path.basename(img_path))
        annotate_and_save(image, results, save_path, classes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory of the onnx model"
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
