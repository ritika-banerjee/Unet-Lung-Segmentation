# Unet-Lung-Segmentation
This project involves implementing a UNet architecture to accurately segment lung regions from medical images, such as X-rays or CT scans. By isolating the lung areas, the model aims to support downstream analysis, disease detection, or assessment of lung conditions. 

# Lung Segmentation using U-Net

This project implements a U-Net model to perform lung segmentation on medical images (e.g., CT scans or X-rays). With convolutional layers and skip connections, the U-Net model accurately segments lung regions, supporting various medical imaging tasks and disease analysis.

## Features
- **Custom Data Loader**: Loads and preprocesses images and corresponding masks.
- **Data Augmentation**: Simple augmentations, such as horizontal flips, to increase data diversity.
- **U-Net Architecture**: Encoder-decoder structure with skip connections for detailed segmentation.
- **Custom IoU Metric**: Uses Intersection over Union (IoU) to evaluate segmentation accuracy.
- **Training Callbacks**: Includes model checkpointing, early stopping, and learning rate reduction.
- **Visualization**: Plots training history and displays sample predictions.

## Project Structure
- `DataLoader` class: Loads images and masks, resizes to target size, and normalizes pixel values.
- `SegmentationGenerator` class: Generates batches of images and masks with optional augmentation.
- `simple_unet` function: Defines a U-Net model with an adjustable input size.
- `iou_metric` function: Calculates the IoU score as a custom evaluation metric.
- `plot_training_history` function: Plots training and validation loss and IoU metrics.
- `predict_and_visualize` function: Predicts masks on the validation set and visualizes results.

## Getting Started

### Prerequisites
- **Python Libraries**: Install the required libraries by running:
  ```bash
  pip install tensorflow numpy opencv-python scikit-learn matplotlib

## Project Setup

### Organize Data
Store images and corresponding mask files in separate directories:
dataset/ ├── images/ └── masks/


### Adjust Paths
Update `image_dir` and `mask_dir` in the main script to point to your dataset directories.

### Run the Project
To execute the project, run the main script. It will:
- Load and preprocess data
- Train the U-Net model with defined callbacks
- Save the best model
- Plot training metrics and visualize sample predictions

```bash
python lung_segmentation.py

