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

def load_mushroom_data(image_size, path, shuffle=True, class_frequency=True):
    """
    Loads mushroom image data from the specified path.

    Args:
        image_size (int): The desired size of the images.
        path (str): The path to the dataset folder.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        class_frequency (bool, optional): Whether to calculate class frequencies. Defaults to True.

    Returns:
        tuple: A tuple containing the image data (X) and labels (Y).
    """
    size = image_size
    X = []
    Y = []
    labels = ['Agaricus', 'Boletus', 'Destroying Angel']
    label_map = {label: i for i, label in enumerate(labels)}

    for i, direct in enumerate(labels):
        files_in_folder = glob.glob(path + '/' + direct + '/*.jpg')
        for file in files_in_folder:
            try:
                data = plt.imread(file)
                data = cv2.resize(data, (size, size))
                data = data.astype('float32') / 255
                X.append(data)
                Y.append(label_map[direct])
            except OSError as e:
                print(f"Error loading image {file}: {e}")
                # You can choose to skip the corrupted image or handle it differently

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
