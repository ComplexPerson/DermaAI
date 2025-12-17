import argparse
import os
import torch
from PIL import Image
import json
import pandas as pd
from dataset import get_transforms
from utils import build_model, find_latest_checkpoint, load_model_state, DISEASE_MAP


def predict_image(model, image_path, image_size, class_names, device, topk: int = 3):
    transform = get_transforms(image_size, is_train=False)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    # top-k
    idxs = probs.argsort()[::-1][:topk]
    results = []
    for i in idxs:
        results.append({"class": class_names[i], "confidence": float(probs[i])})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--output-json", default=None, help="Write JSON results to this file"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--metadata", default="data/HAM10000_metadata.csv")
    parser.add_argument("--backbone", default="resnet34")
    args = parser.parse_args()

    # determine class names from metadata
    md = pd.read_csv(args.metadata)
    class_names = sorted(md["dx"].unique().tolist())
    num_classes = len(class_names)

    ckpt = args.checkpoint or find_latest_checkpoint()
    if ckpt is None:
        print(json.dumps({"error": "no checkpoint found"}))
        return

    model = build_model(num_classes=num_classes, backbone=args.backbone, pretrained=False)
    model = load_model_state(model, ckpt, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_class_names = [DISEASE_MAP.get(name, name) for name in class_names]
    results = predict_image(
        model, args.image, args.image_size, full_class_names, device, topk=args.topk
    )
    out = {"checkpoint": ckpt, "results": results}
    txt = json.dumps(out)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
    print(txt)


if __name__ == "__main__":
    main()