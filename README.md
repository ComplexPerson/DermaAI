# SkinAI: Skin Lesion Classification

This project provides a PyTorch-based pipeline for training a skin lesion classification model on the HAM10000 dataset. It also includes a cross-platform desktop application built with Electron for performing local predictions using a trained ONNX model.

## Project Structure

```
├── app/                  # Electron GUI application
│   ├── main.js           # Main process
│   ├── renderer.js       # Frontend logic
│   └── index.html        # UI layout
├── data/                 # Data directory (place HAM10000 files here)
├── checkpoints/          # Saved model checkpoints and ONNX model
├── dataset.py            # PyTorch dataset and transforms
├── train.py              # Model training script
├── eval.py               # Model evaluation script
├── predict_local.py      # CLI-based prediction script (Python)
├── export_onnx.py        # Script to export PyTorch model to ONNX format
├── utils.py              # Model building and checkpoint helpers
├── check_gpu.py          # Script to verify GPU setup
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Features

- **Training**: Train a deep learning model (e.g., ResNet, Timm models) on the HAM10000 dataset.
- **Prediction**: Get predictions for a local image file from either the command line (Python) or a user-friendly GUI (Electron with ONNX).
- **GUI**: A simple desktop app to select an image and view the top classification results using a pre-exported ONNX model.

---

## Setup and Installation

There are two main components to this project: the Python-based machine learning pipeline and the Node.js-based GUI application.

### Part 1: Python Environment (for Training & CLI Prediction)

This setup is required for training the model, exporting to ONNX, or running predictions from the command line.

**Prerequisites:**
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management.
- An NVIDIA GPU with CUDA for accelerated training.

**Instructions:**

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a conda environment:**
    ```sh
    conda create --name skinai python=3.10
    conda activate skinai
    ```

3.  **Install PyTorch with CUDA:**
    Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and run the command that matches your system's CUDA version. For example:
    ```sh
    # Example for CUDA 12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Part 2: GUI Application (for GUI Prediction)

This setup is required to use the desktop application.

**Prerequisites:**
- [Node.js](https://nodejs.org/) (which includes npm).

**Instructions:**

1.  **Navigate to the `app` directory:**
    ```sh
    cd app
    ```
2.  **Install dependencies:**
    This will download Electron, ONNX Runtime, jimp, and other necessary packages.
    ```sh
    npm install
    ```

---

## Usage

### Training a Model

The `train.py` script is used to train a new model.

```sh
python train.py --data-dir data --metadata data/HAM10000_metadata.csv --epochs 10 --batch-size 32
```
- A trained model checkpoint will be saved in the `checkpoints/` directory.
- Use `--help` to see all available arguments.

### Exporting Model to ONNX

After training a PyTorch model, you can export it to ONNX format for use with the GUI application.

```sh
python export_onnx.py --checkpoint checkpoints/best_model.pth --output checkpoints/model.onnx --backbone resnet34 --image-size 224
```
- Replace `best_model.pth` with the path to your trained PyTorch model checkpoint.
- The exported ONNX model (`model.onnx`) will be saved in the `checkpoints/` directory.

### Prediction (via GUI)

The easiest way to get a prediction is with the desktop app. **Make sure you have exported an ONNX model to `checkpoints/model.onnx` before running the GUI.**

1.  **Start the application:**
    From inside the `app/` directory, run:
    ```sh
    npm start
    ```
2.  **Use the App:**
    - Click "Choose Image" to select a local image file.
    - The image will be previewed in the app.
    - Click "Predict" to run the model. The top predictions will be displayed.

### Prediction (via Command Line)

You can also get a prediction for a single image using the `predict_local.py` script. You must provide a path to a trained model checkpoint.

```sh
python predict_local.py --image /path/to/your/image.jpg --model-path checkpoints/best_model.pth
```

### Verify GPU Setup

The `check_gpu.py` script can be used to quickly confirm that PyTorch is correctly configured to use your NVIDIA GPU.

```sh
python check_gpu.py
```
This will print PyTorch and CUDA details and run a quick test on the GPU.
