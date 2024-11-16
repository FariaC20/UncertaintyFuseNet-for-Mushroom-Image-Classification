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

def load_mushroom_data(image_size=150, path='/content/drive/MyDrive/Mushrooms', shuffle=True, class_frequency=False):
    """
    Loads and preprocesses mushroom image data.

    Args:
        image_size (tuple): Desired size for resizing images (default: (150, 150)).
        path (str): Path to the dataset folder (default: 'Data').
        shuffle (bool): Whether to shuffle the data (default: True).
        class_frequency (bool): Whether to calculate class frequencies (default: False).

    Returns:
        tuple: A tuple containing the preprocessed images (X) and labels (Y).
    """
    X = []
    Y = []
    labels = {'Agaricus': 0, 'Boletus': 1, 'Destroying Angel': 2, 'Entoloma': 3, 'Hygrocybe': 4,
              'Lactarius': 5, 'Russula': 6, 'Suillus': 7}  # Define your class labels here

    for folder in os.listdir(path):
        label = labels.get(folder)  # Get label for the current folder

        if label is not None:  # Skip folders not in the labels dictionary
            folder_path = os.path.join(path, folder)

            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    # Load and resize image using OpenCV
                    img = cv2.imread(img_path)
                    if img is not None:  # Check if image was loaded successfully
                        img = cv2.resize(img, image_size)
                        X.append(img)
                        Y.append(label)
                    else:
                        print(f"Warning: Could not load image: {img_path}") 
                except Exception as e:
                    print(f"Error loading or processing image {img_path}: {e}")
                    # Handle the error appropriately, e.g., skip the image or raise an exception

    # Convert to numpy array outside the loop after all images are processed
    X = np.array(X)  
    Y = np.array(Y)
    if class_frequency:
        print("Class Frequency:", Counter(Y))

    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

    return X, Y
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
