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
    size = image_size
    X = []
    Y = []
    labels = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
    label_dict = {'Agaricus': 0, 'Amanita': 1, 'Boletus': 2, 'Cortinarius': 3, 'Entoloma': 4, 'Hygrocybe': 5, 'Lactarius': 6, 'Russula': 7, 'Suillus': 8}
    for direct in labels:
        files_in_folder = glob.glob(path + '/' + direct + '/*.jpg')
        for file in files_in_folder:
            try:
                data = plt.imread(file)
                data = cv2.resize(data, (size, size))
                data = data.astype('float32') / 255
                X.append(data)
                Y.append(label_dict[direct])
            except OSError as e:
                print(f"Skipping corrupted file: {file} - {e}")  # Print an error message and skip the file

    print(len(X))
    X = np.array(X).astype(float)
    X = transform_images(X)
    X = X[:, :, :, None]

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = np.array(Y).astype(float)
    Y = to_categorical(Y, len(files))

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
