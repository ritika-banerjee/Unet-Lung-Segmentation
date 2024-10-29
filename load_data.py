import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

class datasetLoader:
    def __init__(self, img_preprocessors=None, mask_preprocessor=None):
        self.img_preprocessors = img_preprocessors
        self.mask_preprocessor = mask_preprocessor

    def load(self, img_dir, mask_dir):
        images = []
        masks = []
        print("loading data")
        total_images = len(os.listdir(img_dir))
        with tqdm(total=total_images, desc="Loading Images", unit="Image") as pbar:
            for image_name in os.listdir(img_dir):
                image_path = os.path.join(img_dir, image_name)
                image = cv2.imread(image_path)
                mask_path = os.path.join(mask_dir, image_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if self.img_preprocessors:
                    # Remove the check for list or tuple
                    image = self.img_preprocessors.preprocess_image(image)

                if self.mask_preprocessor:
                    mask = self.mask_preprocessor.preprocess_mask(mask)

                images.append(image)
                masks.append(mask)
                pbar.update(1)

        return np.array(images), np.array(masks)