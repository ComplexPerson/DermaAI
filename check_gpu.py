"""
Simple smoke test to verify PyTorch + CUDA on your machine.
Run this after installing PyTorch to confirm CUDA works and a small model can run on the GPU.
"""

import torch
from torchvision import models


def main():
    print("torch version:", torch.__version__)
    cuda_avail = torch.cuda.is_available()
    print("cuda available:", cuda_avail)
    print("device count:", torch.cuda.device_count())
    if cuda_avail and torch.cuda.device_count() > 0:
        print("device name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if cuda_avail else "cpu")
    # Small model and dummy forward pass
    model = models.resnet18(pretrained=False)
    model.to(device)
    model.eval()

    # small batch on the device
    x = torch.randn(2, 3, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    print("forward pass OK, output shape:", out.shape)

    # test mixed precision if gpu available
    if cuda_avail:
        amp_ok = False
        try:
            with torch.cuda.amp.autocast():
                _ = model(x)
            amp_ok = True
        except Exception as e:
            print("AMP (autocast) test failed:", e)
        print("AMP autocat OK:", amp_ok)


if __name__ == "__main__":
    main()
