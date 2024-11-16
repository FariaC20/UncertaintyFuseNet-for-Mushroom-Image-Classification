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

def load_mushroom_data(image_size=150, path='/content/drive/MyDrive/Mushrooms', shuffle=False, class_frequency=False):
       """
    Load mushroom image data from the specified path.

    Args:
        image_size (int): The desired size to resize the images to.
        path (str): The path to the dataset folder.
        shuffle (bool): Whether to shuffle the data.
        class_frequency (bool): Whether to calculate class frequencies.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The image data and corresponding labels.
    """

    size = image_size
    direct = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe',
              'Lactarius', 'Russula', 'Suillus'] 
    X = []
    Y = []
    for i in range(len(direct)):
        files_in_folder = glob.glob(path + '/' + direct[i] + '/*.jpg')
        for file in files_in_folder:
            try:
                data = plt.imread(file)
                data = cv2.resize(data, (size, size))
                data = data.astype('float32') / 255
                X.append(data)
                Y.append(i)
            except OSError as e:
                print(f"Error reading file {file}: {e}")
                # You can choose to skip the file or handle it differently

    X = np.array(X)
    Y = np.array(Y)
    if shuffle:
        X, Y = shuffle(X, Y, random_state=42)  # Shuffle data

    if class_frequency:
        # Calculate class frequencies
        unique, counts = np.unique(Y, return_counts=True)
        class_frequency = dict(zip(unique, counts))
        print("Class Frequencies:", class_frequency)
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
