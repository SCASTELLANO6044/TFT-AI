import cv2
import numpy as np
import os
import tensorflow as tf
import logging
import sys
# noinspection PyUnresolvedReferences
from model.utils import Utils
from sklearn.model_selection import train_test_split

EPOCHS = 50
IMG_WIDTH = 100
IMG_HEIGHT = 75
MODEL_NAME = "my_model.keras"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'model')
METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files',
                             'HAM10000_metadata.csv')
IMAGES_FOLDER_PART1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files',
                                   'HAM10000_images_part_1')
IMAGES_FOLDER_PART2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files',
                                   'HAM10000_images_part_2')
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'log.log')

if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

logging.basicConfig(format="%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s",
                    datefmt="%H:%M:%S",
                    filename=LOG_FILE_PATH)

logging.getLogger().setLevel(logging.INFO)

CATEGORIES_DICT = {0: 'Enfermedad de Bowen.',
                   1: 'Carcinoma de células basales.',
                   2: 'Dermatofibroma.',
                   3: 'Lesión Vascular.',
                   4: 'Lunar común.',
                   5: 'Queratosis benigna.',
                   6: 'Melanoma.'}
CATEGORIES_MAP = {'akiec': 0, 'bcc': 1, 'df': 2, 'vasc': 3, 'nv': 4, 'bkl': 5, 'mel': 6}


def load_data():
    images, labels = Utils.load_images(METADATA_FILE, IMAGES_FOLDER_PART1, IMAGES_FOLDER_PART2, IMG_WIDTH, IMG_HEIGHT)

    images = Utils.flat_images(images)

    images, labels = Utils.smote_image_generation(images, labels)

    images = Utils.return_images_to_original_dimensions(images, IMG_WIDTH, IMG_HEIGHT)

    labels = np.array(labels)
    images = np.array(images) / 255.0

    labels = tf.keras.utils.to_categorical(labels)

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
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


class Model:

    @staticmethod
    def main(image_path):
        with open(LOG_FILE_PATH, 'w') as log_file:
            sys.stdout = log_file
            try:
                logging.info(msg="Starting to load model from file.")

                model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME))

                logging.info(msg="Finished to load model from file.")

            except OSError:
                logging.info(msg="Model not found in files.")

                logging.info(msg="Starting to load data.")

                labels, images = load_data()

                x_train, x_test, y_train, y_test = train_test_split(
                    images, labels, test_size=0.2, random_state=42
                )

                logging.info("Finished to load data.")

                logging.info("Starting to configure model.")

                model = get_model()

                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5,
                                                                  restore_best_weights=True)

                logging.info("Finished to configure model.")

                logging.info("Starting to train and evaluate model.")

                model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stopping], verbose=2)

                model.evaluate(x_test, y_test, verbose=2)

                logging.info("Finished to train and evaluate model.")

                logging.info("Starting to save the model.")

                model.save(filepath=os.path.join(MODEL_DIR, MODEL_NAME), overwrite=True, save_format="keras")

                logging.info("Finished to save the model.")

            logging.info("Starting to prepare the prediction.")

            img_to_predict = cv2.imread(image_path)  # Replace with the path to your test image
            img_to_predict = cv2.resize(img_to_predict, (IMG_WIDTH, IMG_HEIGHT))
            img_to_predict = np.expand_dims(img_to_predict, axis=0)

            # Make a prediction
            prediction = model.predict(img_to_predict)

            # Get the category with the highest probability
            predicted_category = np.argmax(prediction)

            logging.info("Finished to prepare the prediction.")

            return CATEGORIES_DICT.get(predicted_category)
