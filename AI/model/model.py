import cv2
import numpy as np
import os
import tensorflow as tf
from model.utils import Utils
from sklearn.model_selection import train_test_split

EPOCHS = 50
IMG_WIDTH = 100
IMG_HEIGHT = 75
NUM_CATEGORIES = 7
TEST_SIZE = 0.2
MODEL_NAME = "my_model.keras"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'model')
METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files', 'HAM10000_metadata.csv')
IMAGES_FOLDER_PART1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files', 'HAM10000_images_part_1')
IMAGES_FOLDER_PART2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files', 'HAM10000_images_part_2')

CATEGORIES_DICT = {0: 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease', 1: 'basal cell carcinoma',
                   2: 'dermatofibroma',
                   3: 'vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)',
                   4: 'melanocytic nevi', 5: 'benign keratosis-like lesions', 6: 'melanoma '}
CATEGORIES_MAP = {'akiec': 0, 'bcc': 1, 'df': 2, 'vasc': 3, 'nv': 4, 'bkl': 5, 'mel': 6}


def load_data():

    images, labels = Utils.load_images(METADATA_FILE, IMAGES_FOLDER_PART1, IMAGES_FOLDER_PART2, IMG_WIDTH, IMG_HEIGHT)

    images = Utils.flat_images(images)

    images, labels = Utils.smote_image_generation(images, labels)

    images = Utils.return_images_to_original_dimensions(images, IMG_WIDTH, IMG_HEIGHT)

    labels = np.array(labels)
    images = np.array(images) / 255.0

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

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

            model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stopping], verbose=1)

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
