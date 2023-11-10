import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
DATASET_DIR = os.path.join('.','gtsrb')
MODEL_NAME = "model"
MODEL_DIR = os.path.join('.',MODEL_NAME)

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python.exe TFT-AI.py [input_image]")

    try:
        #Just recover the model in case it is already trained.
        model = tf.keras.models.load_model(MODEL_DIR)
    except:
        # Get image arrays and labels for all image files
        images, labels = load_data(DATASET_DIR)

        # Split data into training and testing sets
        labels = tf.keras.utils.to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )

        # Get a compiled neural network
        model = get_model()

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance
        model.evaluate(x_test, y_test, verbose=2)

        #Save the model
        model.save(MODEL_NAME)
        print(f"Model saved to {MODEL_NAME}.")

    img_to_predict = cv2.imread(sys.argv[1])  # Replace with the path to your test image
    img_to_predict = cv2.resize(img_to_predict, (IMG_WIDTH, IMG_HEIGHT))
    img_to_predict = np.expand_dims(img_to_predict, axis=0)

    # Make a prediction
    prediction = model.predict(img_to_predict)

    # Get the category with the highest probability
    predicted_category = np.argmax(prediction)

    print(f"Predicted category: {predicted_category}")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    clean_image_list = []
    label_list = []

    for folder in os.listdir(data_dir):
        print("Processing folder: " + data_dir + os.sep + folder)
        for file_name in os.listdir(os.path.join(data_dir, folder)):
            raw_image = cv2.imread(os.path.join(data_dir, folder, file_name))
            clean_image = cv2.resize(raw_image, (IMG_WIDTH, IMG_HEIGHT))

            clean_image_list.append(clean_image)
            label_list.append(folder)

    return clean_image_list, label_list


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            NUM_CATEGORIES,
            activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
