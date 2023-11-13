import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import csv

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 100
IMG_HEIGHT = 75
NUM_CATEGORIES = 7
TEST_SIZE = 0.4
MODEL_NAME = "model"
MODEL_DIR = os.path.join('.', MODEL_NAME)
METADATA_FILE = os.path.join('.', 'dataverse_files', 'HAM10000_metadata.csv')
IMAGES_FOLDER_PART1 = os.path.join('.', 'dataverse_files', 'HAM10000_images_part_1')
IMAGES_FOLDER_PART2 = os.path.join('.', 'dataverse_files', 'HAM10000_images_part_2')

CATEGORIES_DICT = {}
CATEGORIES_DICT[0] = 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease'
CATEGORIES_DICT[1] = 'basal cell carcinoma'
CATEGORIES_DICT[2] = 'dermatofibroma'
CATEGORIES_DICT[3] = 'vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)'
CATEGORIES_DICT[4] = 'melanocytic nevi'  ##############################
CATEGORIES_DICT[5] = 'benign keratosis-like lesions'
CATEGORIES_DICT[6] = 'melanoma '


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python.exe TFT-AI.py [input_image]")

    try:
        # Just recover the model in case it is already trained.
        model = tf.keras.models.load_model(MODEL_DIR)
    except:
        # Get image arrays and labels for all image files
        images, labels = load_data()

        # Split data into training and testing sets
        labels = tf.keras.utils.to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )

        model = get_model()

        model.fit(x_train, y_train, epochs=EPOCHS)

        model.evaluate(x_test, y_test, verbose=2)

        model.save(MODEL_NAME)
        print(f"Model saved to {MODEL_NAME}.")

    img_to_predict = cv2.imread(sys.argv[1])  # Replace with the path to your test image
    img_to_predict = cv2.resize(img_to_predict, (IMG_WIDTH, IMG_HEIGHT))
    img_to_predict = np.expand_dims(img_to_predict, axis=0)

    # Make a prediction
    prediction = model.predict(img_to_predict)

    # Get the category with the highest probability
    predicted_category = np.argmax(prediction)

    print(f"Predicted category: {CATEGORIES_DICT.get(predicted_category)}")


def load_data():
    metadata_dict = {}
    clean_image_list = []
    label_list = []

    with open(METADATA_FILE, 'r', newline='') as data:
        reader = csv.reader(data)
        next(reader, None)

        for line in reader:
            image_id = line[1]
            lesion_type = line[2]
            match lesion_type:
                case 'akiec':
                    metadata_dict[image_id] = 0
                case 'bcc':
                    metadata_dict[image_id] = 1
                case 'df':
                    metadata_dict[image_id] = 2
                case 'vasc':
                    metadata_dict[image_id] = 3
                case 'nv':
                    metadata_dict[image_id] = 4
                case 'bkl':
                    metadata_dict[image_id] = 5
                case 'mel':
                    metadata_dict[image_id] = 6

    print("Start to process the 1st part of the dataset")
    for file_name in os.listdir(IMAGES_FOLDER_PART1):
        raw_image = cv2.imread(os.path.join(IMAGES_FOLDER_PART1, file_name))
        clean_image = cv2.resize(raw_image, (IMG_WIDTH, IMG_HEIGHT))
        clean_image_list.append(clean_image)
        label_list.append(metadata_dict.get(file_name[:-4]))

    print("Start to process the 2nd part of the dataset")
    for file_name in os.listdir(IMAGES_FOLDER_PART2):
        raw_image = cv2.imread(os.path.join(IMAGES_FOLDER_PART2, file_name))
        clean_image = cv2.resize(raw_image, (IMG_WIDTH, IMG_HEIGHT))
        clean_image_list.append(clean_image)
        label_list.append(metadata_dict.get(file_name[:-4]))

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
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),

        tf.keras.layers.Dropout(0.05),

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
