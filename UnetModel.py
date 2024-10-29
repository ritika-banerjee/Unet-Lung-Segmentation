import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Data loading and preprocessing
class DataLoader:
    def __init__(self, image_dir, mask_dir, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size

    def load_data(self):
        image_files = os.listdir(self.image_dir)
        images = []
        masks = []

        for img_file in image_files:
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, img_file)

            if os.path.exists(mask_path):
                img = self.load_image(img_path)
                mask = self.load_mask(mask_path)

                images.append(img)
                masks.append(mask)

        return np.array(images), np.array(masks)

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        return img / 255.0

    def load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size)
        mask = (mask > 0).astype(np.float32)
        return np.expand_dims(mask, axis=-1)


# Data generator
class SegmentationGenerator(Sequence):
    def __init__(self, images, masks, batch_size=32, augment=False):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.masks[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.augment:
            # Simple augmentation (flip horizontally)
            for i in range(len(batch_images)):
                if np.random.rand() < 0.5:
                    batch_images[i] = np.fliplr(batch_images[i])
                    batch_masks[i] = np.fliplr(batch_masks[i])

        return np.array(batch_images), np.array(batch_masks)


# Simple U-Net model
def simple_unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv3])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv2])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv1])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# IoU metric
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())


# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['iou_metric'], label='Training IoU')
    plt.plot(history.history['val_iou_metric'], label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Predict and visualize results
def predict_and_visualize(model, images, masks, num_samples=3):
    predictions = model.predict(images)

    for i in range(num_samples):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i, :, :, 0], cmap='gray')
        plt.title('True Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# Main execution
if __name__ == "__main__":
    # Set paths
    image_dir = r"D:\Segment\Segmentation\dataset\images"
    mask_dir = r"D:\Segment\Segmentation\dataset\masks"

    # Load and preprocess data
    data_loader = DataLoader(image_dir, mask_dir)
    images, masks = data_loader.load_data()

    # Split data
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42)

    # Create data generators
    train_gen = SegmentationGenerator(train_images, train_masks, batch_size=32, augment=True)
    val_gen = SegmentationGenerator(val_images, val_masks, batch_size=32, augment=False)

    # Create and compile model
    model = simple_unet()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[iou_metric])

    # Set up callbacks
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_iou_metric', mode='max')
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)

    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Plot training history
    plot_training_history(history)

    # Save final model
    model.save('final_model.keras')

    # Predict and visualize results
    predict_and_visualize(model, val_images, val_masks)

    print("Training completed.")