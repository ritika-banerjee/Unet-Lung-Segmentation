# Unet-Lung-Segmentation
This project involves implementing a UNet architecture to accurately segment lung regions from medical images, such as X-rays or CT scans. By isolating the lung areas, the model aims to support downstream analysis, disease detection, or assessment of lung conditions. 

Lung Segmentation using U-Net
This project implements a U-Net model to perform lung segmentation on medical images (e.g., CT scans or X-rays). Using convolutional layers and skip connections, the U-Net model accurately segments the lung regions, which can assist in various medical imaging tasks.

Features
Custom Data Loader: Loads and preprocesses images and their corresponding masks.
Data Augmentation: Basic augmentation such as horizontal flips to increase training data diversity.
U-Net Architecture: Encoder-decoder structure with skip connections for precise boundary segmentation.
Custom IoU Metric: Intersection over Union (IoU) metric to evaluate segmentation accuracy.
Training Callbacks: Includes model checkpointing, early stopping, and learning rate reduction.
Visualization: Plots training history and sample predictions.
Project Structure
DataLoader class: Loads images and masks, resizes them to a standard target size, and normalizes pixel values.
SegmentationGenerator class: Generates batches of images and masks with optional augmentation.
simple_unet function: Defines a U-Net model with a configurable input size.
iou_metric function: Calculates IoU score as a custom evaluation metric.
plot_training_history function: Plots training and validation loss and IoU metrics.
predict_and_visualize function: Predicts segmentation masks on the validation set and displays results.
Getting Started
Prerequisites
Python libraries: Install necessary libraries, e.g.,
bash
Copy code
pip install tensorflow numpy opencv-python scikit-learn matplotlib
Project Setup
Organize Data: Store images and corresponding mask files in separate directories.
Copy code
dataset/
├── images/
└── masks/
Adjust Paths: Update image_dir and mask_dir in the main code to point to your data directories.
Run the Project
To execute the project, run the main code. It will:

Load and preprocess data
Train the U-Net model with defined callbacks
Save the best model
Plot training metrics and visualize sample predictions
bash
Copy code
python lung_segmentation.py
Results
The script will output:

A saved model file (best_model.keras and final_model.keras)
Plots for training and validation metrics
Sample predictions showing original images, true masks, and predicted masks





