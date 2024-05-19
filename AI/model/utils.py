import os.path
import warnings
import config.config as cfg
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from imblearn import over_sampling
from sklearn.metrics import confusion_matrix
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)


class Utils:

    @staticmethod
    def load_images(metadata_file, images_folder_part1, images_folder_part2, img_width, img_height):
        metadata_dict = {}
        clean_image_list = []
        label_list = []

        with open(metadata_file, 'r', newline='') as data:
            reader = csv.reader(data)
            next(reader, None)

            for line in reader:
                image_id = line[1]
                lesion_type = line[2]
                metadata_dict[image_id] = cfg.categories_map[lesion_type]

        for folder_path in [images_folder_part1, images_folder_part2]:
            for file_name in os.listdir(folder_path):
                raw_image = cv2.imread(os.path.join(folder_path, file_name))
                clean_image = cv2.resize(raw_image, (img_width, img_height))
                clean_image_list.append(clean_image)
                label_list.append(metadata_dict.get(file_name[:-4]))

        return clean_image_list, label_list

    @staticmethod
    def flat_images(images):
        flat_images = []

        for arr in images:
            new_arr = arr.reshape(-1)
            flat_images.append(new_arr)

        return flat_images

    @staticmethod
    def smote_image_generation(images, labels):
        oversample = over_sampling.SMOTE()
        images, labels = oversample.fit_resample(images, labels)

        return images, labels

    @staticmethod
    def return_images_to_original_dimensions(images, width, height):
        original_dimensions = (height, width, 3)

        image_list = [np.array(arr).reshape(original_dimensions) for arr in images]

        return image_list

    @staticmethod
    def collect_training_results(history, x_test, y_test, model):

        Utils.loss_training_metric(history)
        Utils.accuracy_training_metric(history)
        Utils.confusion_matrix(x_test, y_test, model)

    @staticmethod
    def loss_training_metric(history):

        plt.title("Pérdida durante el entrenamiento.")
        plt.xlabel('Época.')
        plt.plot(history.history["loss"])

        plt.savefig(cfg.loss_training_histogram_file)

    @staticmethod
    def accuracy_training_metric(history):

        plt.title("Precisión durante el entrenamiento.")
        plt.xlabel('Época.')
        plt.plot(history.history["accuracy"])

        plt.savefig(cfg.accuracy_training_histogram_file)

    @staticmethod
    def confusion_matrix(x_test, y_test, model):
        y_pred = model.predict(x_test)

        y_pred_classes = np.argmax(y_pred, axis=1)

        y_true = np.argmax(y_test, axis=1)

        confusion_mtx = confusion_matrix(y_true, y_pred_classes)

        _, ax = plt.subplots(figsize=(8, 8))

        sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        plt.savefig(cfg.confusion_matrix_file)
