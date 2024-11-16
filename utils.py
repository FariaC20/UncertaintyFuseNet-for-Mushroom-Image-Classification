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


def load_mushroom_data(image_size, path, shuffle=True, class_frequency=False):
    """
    Loads mushroom image data from the specified path.

    Args:
        image_size (int): The desired size for resizing images.
        path (str): The path to the dataset folder.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        class_frequency (bool, optional): Whether to adjust class weights based on frequency. Defaults to False.

    Returns:
        tuple: A tuple containing the image data (X) and labels (Y).
    """
    size = image_size
    X = []
    Y = []
    class_labels = {'Agaricus': 0, 'Boletus': 1, 'DestroyingAngel': 2, 'Entoloma': 3,
                     'Lactarius': 4, 'Pluteus': 5, 'Russula': 6, 'Suillus': 7}

    for direct in os.listdir(path):
        files_in_folder = glob.glob(path + '/' + direct + '/*.jpg')
        for file in files_in_folder:
            try:
                data = plt.imread(file)
                data = cv2.resize(data, (size, size))
                data = data.astype('float32') / 255
                X.append(data)
                Y.append(class_labels[direct]) 
            except OSError as e:
                print(f"Skipping corrupted image: {file} due to error: {e}")
                continue  # Skip to the next image

    X = np.array(X)
    Y = np.array(Y)

    if shuffle:
        from sklearn.utils import shuffle
        X, Y = shuffle(X, Y, random_state=42)  # Shuffle data

    if class_frequency:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', np.unique(Y), Y)
        class_weights = dict(enumerate(class_weights))
        print("Class Weights:", class_weights)
        return X, Y, class_weights 

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
