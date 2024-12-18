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
from sklearn.utils import shuffle

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def transform_images(images: np.ndarray):
    """
    Transform images to [-1, 1]
    """
    images = 2 * images.astype(np.float32) - 1
    return images

def load_mushroom_data(image_size=150, path='/content/drive/MyDrive/Mushrooms', shuffle=False, class_frequency=False):
    size = image_size
    files = listdir(path)
    X = []
    Y = []

    for direct in files:
        files_in_folder = glob.glob(path + '/' + direct + '/*.*')
        for file in files_in_folder:
            try:
                data = plt.imread(file)
                data = cv2.resize(data, (size, size))
                data = data.astype('float32') / 255
                if len(data.shape) > 2 and data.shape[2] == 3:
                    data = rgb2gray(data)
                if len(data.shape) > 2 and data.shape[2] == 4:
                    data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    data = rgb2gray(data)
                X.append(data)
                Y.append(direct)
            except OSError as e:
                print(f"Skipped file {file} due to error: {e}")
            except Exception as e:
                print(f"Unexpected error occurred with file {file}: {e}")

    print(f"Total valid images loaded: {len(X)}")
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
          # Plotting with improved readability
        plt.figure(figsize=(10, 6))  # Adjust the figure size for better visibility
        plt.bar(unique, counts)
        plt.title('Class Frequency (Percent)')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels for readability
        plt.tight_layout()  # Ensure the layout fits well within the figure
        plt.show()
        #plt.bar(unique, counts)
        #plt.title('Class Frequency (Percent)')
        #plt.xlabel('Class')
        #plt.ylabel('Frequency')
        #plt.show()
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
