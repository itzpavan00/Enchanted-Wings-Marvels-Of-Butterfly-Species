import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

# Define paths
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
img_size = (224, 224)  # ResNet50 input size

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to split dataset (if not already split)
def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.endswith('.jpg')]
        
        # Split into train+val and test
        train_val_imgs, test_imgs = train_test_split(images, test_size=1-(train_ratio+val_ratio), random_state=42)
        # Split train+val into train and val
        train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
        
        # Copy images to respective directories
        for split, split_dir in [(train_imgs, train_dir), (val_imgs, val_dir), (test_imgs, test_dir)]:
            cls_split_dir = os.path.join(split_dir, cls)
            os.makedirs(cls_split_dir, exist_ok=True)
            for img in split:
                shutil.copy(img, cls_split_dir)

# Run split if dataset is in a single folder (uncomment if needed)
# source_dir = 'path_to_unsplit_dataset'
# split_dataset(source_dir)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Save class indices
class_indices = train_generator.class_indices
np.save('class_indices.npy', class_indices)

print("Dataset prepared: Training, validation, and test sets loaded with augmentation.")