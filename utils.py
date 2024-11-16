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

def load_mushroom_data(image_size=150, path='/content/drive/MyDrive/Mushrooms', shuffle=True, class_frequency=True):
     """
    Loads mushroom image data from the specified path.

    Args:
        image_size (int): Desired size for resizing images.
        path (str): Path to the dataset directory.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        class_frequency (bool, optional): Whether to calculate class frequencies. Defaults to True.

    Returns:
        tuple: A tuple containing the image data and labels.
    """
    X = []  
    y = []
    labels = {}
    current_label = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Adjust file extensions if necessary
                file_path = os.path.join(root, file)
                
                # Use Pillow to open and resize the image ensuring consistency
                image = Image.open(file_path)
                image = image.resize((image_size, image_size))
                image_np = np.array(image)
                
                # Check if the image has 3 channels (RGB)
                if image_np.ndim != 3 or image_np.shape[2] != 3:
                    # If not, convert it to RGB
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                
                X.append(image_np)


                label_name = os.path.basename(os.path.dirname(file_path))
                if label_name not in labels:
                    labels[label_name] = current_label
                    current_label += 1
                Y.append(labels[label_name])

    # Convert to NumPy arrays after ensuring all images have the same dimensions
    X = np.array(X)
    Y = np.array(Y)

    if shuffle:
        idx = np.random.choice(len(X), size=len(X), replace=False)
        X = X[idx, :, :]
        Y = Y[idx, :]
    if class_frequency:
        classes = le.inverse_transform(np.argmax(Y, axis=1).astype(int))
        unique, counts = np.unique(classes, return_counts=True)
        counts = np.array(counts)
        plt.bar(unique, counts)
        plt.title('Class Frequency(Percent)')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.show()
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
