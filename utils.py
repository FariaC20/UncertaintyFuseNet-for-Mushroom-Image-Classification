import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import cv2
from os import listdir
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os


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
    Load and preprocess mushroom classification dataset.
    """
    size = image_size
    categories = [d for d in listdir(path) if os.path.isdir(os.path.join(path, d))]  # Each folder represents a class
    X = []
    Y = []

    for category in categories:
        class_folder = glob.glob(os.path.join(path, category, '*.jpg'))  # Adjust for dataset images
        for img_path in class_folder:
            try:
                # Load image
                image = cv2.imread(img_path)

                # Convert grayscale to RGB if needed
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # Resize to uniform dimensions
                image = cv2.resize(image, (size, size))

                # Normalize to [0, 1]
                image = image.astype('float32') / 255.0

                X.append(image)
                Y.append(category)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

    print(f"Total images loaded: {len(X)}")

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)

    # Normalize images to [-1, 1]
    X = transform_images(X)

    # Encode class labels
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = np.array(Y)
    Y = to_categorical(Y, num_classes=len(categories))

    # Shuffle data if required
    if shuffle:
        idx = np.random.permutation(len(X))
        X = X[idx]
        Y = Y[idx]

    # Plot class frequencies if required
    if class_frequency:
        class_labels = le.inverse_transform(np.argmax(Y, axis=1))
        unique, counts = np.unique(class_labels, return_counts=True)
        plt.bar(unique, counts)
        plt.title("Class Frequency (Percent)")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()

    return X, Y, categories


def create_dataset(X, Y, batch_size=32, test_size=0.3, random_state=42):
    """
    Split data into training and testing datasets and create TensorFlow datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    validation_dataset = validation_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Specify dataset folder path
    dataset_path = '/content/drive/MyDrive/Mushrooms'  # Replace with the actual dataset folder path from Kaggle
    
    # Load and preprocess data
    X, Y, categories = load_mushroom_data(image_size=150, path=dataset_path, shuffle=True, class_frequency=True)
