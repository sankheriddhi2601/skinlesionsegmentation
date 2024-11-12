import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from metrics import dice_loss, dice_coef, iou
from model import build_multi_resunet

# Adjusted directories based on the correct structure
data_dir ='/Users/Nitin/OneDrive/Desktop/MSCPROJECTSKIN01/ISIC'
train_image_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_Data')
train_mask_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_GroundTruth/ISBI2016_ISIC_Part1_Training_GroundTruth')
test_image_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Test_Data/ISBI2016_ISIC_Part1_Test_Data')
test_mask_dir = os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Test_GroundTruth/ISBI2016_ISIC_Part1_Test_GroundTruth')

# List all files in the directories
train_image_files = sorted([os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
train_mask_files = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.lower().endswith('.png')])
test_image_files = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
test_mask_files = sorted([os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir) if f.lower().endswith('.png')])

print(f"Found {len(train_image_files)} training images and {len(train_mask_files)} training masks.")
print(f"Found {len(test_image_files)} test images and {len(test_mask_files)} test masks.")

def load_and_resize_images(image_files, size):
    images = []
    for img_file in image_files:
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
    return np.array(images)

def load_and_resize_masks(mask_files, size):
    masks = []
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, size)
            mask = mask / 255.0  # Normalize masks to [0, 1]
            masks.append(mask)
    return np.expand_dims(np.array(masks), axis=-1)  # Add channel dimension

# Define the target image size
image_size = (256, 256)

# Load and resize training and test images and masks
train_images_resized = load_and_resize_images(train_image_files, image_size)
train_masks_resized = load_and_resize_masks(train_mask_files, image_size)
test_images_resized = load_and_resize_images(test_image_files, image_size)
test_masks_resized = load_and_resize_masks(test_mask_files, image_size)

# Normalize images
train_images_resized = train_images_resized / 255.0
test_images_resized = test_images_resized / 255.0

print(f"Training images resized shape: {train_images_resized.shape}")
print(f"Training masks resized shape: {train_masks_resized.shape}")
print(f"Test images resized shape: {test_images_resized.shape}")
print(f"Test masks resized shape: {test_masks_resized.shape}")

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_images_resized, train_masks_resized, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(test_images_resized)}")

# Data augmentation
data_gen_args = dict(horizontal_flip=True,
                     vertical_flip=True,
                     rotation_range=90,
                     zoom_range=0.2,
                     shear_range=0.2,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_datagen.fit(X_train, augment=True, seed=42)
mask_datagen.fit(y_train, augment=True, seed=42)

train_image_generator = image_datagen.flow(X_train, batch_size=16, seed=42)
train_mask_generator = mask_datagen.flow(y_train, batch_size=16, seed=42)
train_generator = zip(train_image_generator, train_mask_generator)

# Build the model
model = build_multi_resunet((256, 256, 3))

# Compile the model with Dice loss and Dice coefficient as a metric
model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coef, 'accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
callbacks = [checkpoint, reduce_lr, early_stopping]

# Fit the model
history = model.fit(
    train_generator,
    epochs=50,  # Increase the number of epochs
    steps_per_epoch=len(X_train) // 16,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Evaluate the model
loss, dice_coef, accuracy = model.evaluate(test_images_resized, test_masks_resized)
print(f'Test Loss: {loss}')
print(f'Test Dice Coefficient: {dice_coef}')
print(f'Test Accuracy: {accuracy}')

# Make predictions on the entire test set
predictions = model.predict(test_images_resized)

# Select 5 random indices from the test set
import random
import matplotlib.pyplot as plt

random_indices = random.sample(range(len(test_images_resized)), 5)

# Plot the results
fig, axes = plt.subplots(5, 3, figsize=(15, 15))

for i, idx in enumerate(random_indices):
    # Original image
    axes[i, 0].imshow(test_images_resized[idx])
    axes[i, 0].set_title("Test Image")
    axes[i, 0].axis("off")

    # Ground truth mask
    axes[i, 1].imshow(test_masks_resized[idx].squeeze(), cmap='gray')
    axes[i, 1].set_title("Ground Truth Mask")
    axes[i, 1].axis("off")

    # Predicted mask
    predicted_mask = (predictions[idx] > 0.5).astype(np.uint8)
    axes[i, 2].imshow(predicted_mask.squeeze(), cmap='gray')
    axes[i, 2].set_title("Predicted Mask")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

