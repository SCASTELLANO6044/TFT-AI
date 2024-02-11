import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import csv
import time
import pandas as pd
from glob import glob
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

EPOCHS = 40
IMG_WIDTH = 100
IMG_HEIGHT = 75
NUM_CATEGORIES = 7
TEST_SIZE = 0.2
MODEL_NAME = "my_model.keras"
MODEL_DIR = os.path.join('.', 'data', 'model')
METADATA_FILE = os.path.join('.', 'data', 'dataverse_files', 'HAM10000_metadata.csv')
IMAGES_FOLDER_PART1 = os.path.join('.', 'data', 'dataverse_files', 'HAM10000_images_part_1')
IMAGES_FOLDER_PART2 = os.path.join('.', 'data', 'dataverse_files', 'HAM10000_images_part_2')

CATEGORIES_DICT = {0: 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease', 1: 'basal cell carcinoma',
                   2: 'dermatofibroma',
                   3: 'vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)',
                   4: 'melanocytic nevi', 5: 'benign keratosis-like lesions', 6: 'melanoma '}
CATEGORIES_MAP = {'akiec': 0, 'bcc': 1, 'df': 2, 'vasc': 3, 'nv': 4, 'bkl': 5, 'mel': 6}


def load_data():
    print("Start loading data...")
    start_time = time.time()
    skin_df = pd.read_csv(os.path.join('.', 'data', 'dataverse_files', 'HAM10000_metadata.csv'))
    image_path = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join('data', 'dataverse_files', '*', '*.jpg'))
    }
    skin_df['path'] = skin_df['image_id'].map(image_path.get)
    # Use the path to read images.
    skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((IMG_WIDTH, IMG_HEIGHT))))
    skin_df['dx'] = skin_df['dx'].replace(CATEGORIES_MAP)

    end_time = time.time()
    elapsed_time = end_time - start_time

    labels = skin_df['dx'].tolist()
    images = skin_df['image'].tolist()

    labels = LabelEncoder().fit_transform(labels)

    flat_images = []
    for arr in images:
        new_arr = arr.reshape(-1)
        flat_images.append(new_arr)

    oversample = SMOTE()
    images, labels = oversample.fit_resample(flat_images, labels)

    dimensiones_originales = (IMG_HEIGHT, IMG_WIDTH, 3)

    lista_dimensiones_originales = [np.array(arr).reshape(dimensiones_originales) for arr in images]

    labels = np.array(labels)
    images = np.array(lista_dimensiones_originales) / 255.0

    print("Time passed:", str(elapsed_time / 60)[-2:], "minutes")
    return labels, images


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_dense_net121():
    base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False,
                                                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # Freeze the layers of the pre-trained Xception model
    # for layer in base_model.layers:
    #     layer.trainable = False

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_xception_model():
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False,
                                                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # Freeze the layers of the pre-trained Xception model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class Model:

    @staticmethod
    def main(image_path):

        try:
            # Just recover the model in case it is already trained.
            model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME))
        except OSError:
            # Get image arrays and labels for all image files
            labels, images = load_data()

            # Split data into training and testing sets
            labels = tf.keras.utils.to_categorical(labels)
            x_train, x_test, y_train, y_test = train_test_split(
                images, labels, test_size=TEST_SIZE, random_state=42
            )

            model = get_model()

            model.fit(x_train, y_train, epochs=EPOCHS)

            model.evaluate(x_test, y_test, verbose=2)

            model.save(filepath=os.path.join(MODEL_DIR, MODEL_NAME), overwrite=True, save_format="keras")
            print(f"Model saved to {MODEL_DIR}.")

        img_to_predict = cv2.imread(image_path)  # Replace with the path to your test image
        img_to_predict = cv2.resize(img_to_predict, (IMG_WIDTH, IMG_HEIGHT))
        img_to_predict = np.expand_dims(img_to_predict, axis=0)

        # Make a prediction
        prediction = model.predict(img_to_predict)

        # Get the category with the highest probability
        predicted_category = np.argmax(prediction)

        return CATEGORIES_DICT.get(predicted_category)
