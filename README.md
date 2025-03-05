<div align="center">

<h1 align="center">YOLOv5 Drowsiness Detector</h1>

  <p align="center">
    Detects drowsiness using YOLOv5, trained on a custom dataset.
    <br />
  </p>
</div>

## Table of Contents

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#architecture">Architecture</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#training">Training</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This project implements a drowsiness detection system using YOLOv5, a state-of-the-art object detection model. The model is trained on a custom dataset of images depicting "awake" and "drowsy" states, enabling it to identify signs of drowsiness in real-time video feeds. The project includes a Jupyter Notebook (`drowsiness-detection.ipynb`) that demonstrates the setup, detection process, and training procedure.

### Key Features

*   **Real-time Drowsiness Detection:** Utilizes OpenCV to capture video from a webcam and YOLOv5 to detect drowsiness in each frame.
*   **Custom Dataset Training:** Provides a framework for collecting and labeling custom image datasets for training the drowsiness detection model.
*   **Transfer Learning:** Leverages pre-trained YOLOv5 weights for faster and more efficient training on the custom drowsiness dataset.
*   **Jupyter Notebook Implementation:** Offers a step-by-step guide to setting up the environment, loading the model, performing detections, and training a new model.
*   **Customizable Training Parameters:** Allows users to adjust training parameters such as image size, batch size, and number of epochs.

## Architecture

The drowsiness detection system is built upon the following architecture:

*   **Data Acquisition:** Captures video frames from a webcam using OpenCV.
*   **Object Detection:** Employs a YOLOv5 model to identify faces and classify them as "awake" or "drowsy".
*   **Model Training:** Uses a custom dataset and transfer learning to fine-tune the YOLOv5 model for drowsiness detection.
*   **Real-time Analysis:** Displays the video feed with bounding boxes around detected faces, indicating their drowsiness state.

The core components of the system include:

*   **YOLOv5:** A pre-trained object detection model used as the base for drowsiness detection.
*   **OpenCV:** A library for real-time computer vision, used for capturing and displaying video frames.
*   **PyTorch:** A deep learning framework used for training and running the YOLOv5 model.

## Getting Started

To get started with this project, follow the instructions below:

### Prerequisites

*   Python 3.7 or higher
*   CUDA-enabled GPU (recommended for faster training and inference)
*   Required Python packages:

    ```sh
    pip install torch torchvision torchaudio opencv-python matplotlib numpy
    ```

### Installation

1.  Clone the repository:

    ```sh
    git clone https://github.com/surjahead/yolov5-drowsiness-detector.git
    cd yolov5-drowsiness-detector
    ```

2.  Download the YOLOv5 weights:

    *   The `yolov5s.pt` file is included in the repository. If you need to re-download it, you can do so using the following:
    ```python
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    torch.save(model.state_dict(), 'yolov5s.pt')
    ```

### Usage

1.  **Run the Drowsiness Detection Notebook:**

    *   Open `drowsiness-detection.ipynb` in Jupyter Notebook or Google Colab.
    *   Execute the cells in the notebook sequentially to set up the environment, load the model, and perform real-time drowsiness detection using your webcam.

2.  **Realtime Detection:**

    *   The notebook contains code to capture video from your webcam and display the drowsiness detection results in real-time.
    *   Press `q` to quit the real-time detection.

## Training

To train the model from scratch or fine-tune it with your own dataset:

1.  **Prepare your dataset:**

    *   Collect images of "awake" and "drowsy" faces.
    *   Label the images using a tool like `labelImg` (included in the repository).
    *   Organize the images and labels into the appropriate directories (`data/images` and `data/labels`).
    *   Create a `dataset.yaml` file specifying the paths to your training and validation data, as well as the number of classes and class names.

2.  **Train the Model:**

    *   Navigate to the `yolov5` directory:
    ```sh
    cd yolov5
    ```
    *   Run the training script:
    ```sh
    python train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt
    ```
    *   Adjust the training parameters as needed, such as `--img` (image size), `--batch` (batch size), and `--epochs` (number of epochs).

## Acknowledgments

- This README was created using [gitreadme.dev](https://gitreadme.dev) — an AI tool that looks at your entire codebase to instantly generate high-quality README files.
