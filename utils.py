import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
import cv2
from os import listdir
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import Counter

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def transform_images(images: np.ndarray):
    """
    Transform images to [-1, 1]
    """
    images = 2 * images.astype(np.float32) - 1
    return images


def load_mushroom_data(image_size, path, shuffle, class_frequency):
    """
    Load and preprocess mushroom image data.

    Args:
        image_size (int): The desired size for resizing the images.
        path (str): The path to the dataset folder.
        shuffle (bool): Whether to shuffle the data.
        class_frequency (bool): Whether to calculate class frequencies.

    Returns:
        tuple: A tuple containing the preprocessed images (X) and labels (Y).
    """
    X = []
    Y = []
    classes = os.listdir(path)
    num_classes = len(classes)

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(path, class_name)
        image_files = os.listdir(class_path)

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            try:
                # Open the image using Pillow library
                image = Image.open(image_path)
                
                # Resize the image while preserving aspect ratio
                image.thumbnail((image_size, image_size)) 
                
                # Pad the image if necessary to ensure consistent dimensions
                image_array = np.array(image)
                
                # Calculate padding for each dimension
                pad_height = image_size - image_array.shape[0]
                pad_width = image_size - image_array.shape[1]
                
                # Apply padding to the image
                image_array = np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
                
                X.append(image_array)
                Y.append(class_index)  # Assuming class_index is the label
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue  # Skip to the next image

    X = np.array(X)
    Y = np.array(Y)
    if shuffle:
        from sklearn.utils import shuffle
        X, Y = shuffle(X, Y, random_state=42)  # Shuffle data

    if class_frequency:
        from sklearn.utils.class_weight import compute_class_weight
        # Convert Y to a 1D array of integer labels
        Y_labels = np.array([y for sublist in Y for y in sublist])  
        class_weights = compute_class_weight('balanced', np.unique(Y_labels), Y_labels)
        class_weights = dict(enumerate(class_weights))
        print("Class Weights:", class_weights) 

    return X, Y  # Return data and labels

def create_dataset(X, Y, batch_size):
    np.random.seed(0)
    random.seed(0)
    idx = np.random.choice(len(X), size=len(X), replace=False)
    X = X[idx, :, :, :]
    Y = Y[idx, :]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.cache()
    validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, validation_dataset, X_train, X_test, y_train, y_test
