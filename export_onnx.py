import argparse
import os
import torch
from utils import build_model, load_model_state
from dataset import Ham10000Dataset
import pandas as pd


def export_model_to_onnx(
    checkpoint_path,
    output_path,
    num_classes,
    backbone="resnet34",
    image_size=224,
    opset_version=11,
):
    """
    Exports a PyTorch model to ONNX format.

    Args:
        checkpoint_path (str): Path to the PyTorch model checkpoint.
        output_path (str): Path where the ONNX model will be saved.
        num_classes (int): Number of output classes for the model.
        backbone (str): Backbone architecture of the model (e.g., 'resnet34').
        image_size (int): Input image size (e.g., 224 for 224x224).
        opset_version (int): ONNX opset version to use for export.
    """
    model = build_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model = load_model_state(model, checkpoint_path, map_location="cpu")
    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model successfully exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export a PyTorch model checkpoint to ONNX format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the input PyTorch model checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output ONNX model (.onnx).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        help="Backbone architecture of the model (e.g., 'resnet34', 'resnet50').",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size for the model (e.g., 224 for 224x224).",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version to use for export.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/HAM10000_metadata.csv",
        help="Path to the metadata CSV file to determine number of classes.",
    )

    args = parser.parse_args()

    # Determine num_classes from metadata
    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    md = pd.read_csv(args.metadata)
    class_names = sorted(md["dx"].unique().tolist())
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes from metadata.")

    export_model_to_onnx(
        args.checkpoint,
        args.output,
        num_classes,
        args.backbone,
        args.image_size,
        args.opset_version,
    )


if __name__ == "__main__":
    main()
