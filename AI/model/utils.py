import os.path
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

class Utils:

    @staticmethod
    def prepare_dataframe(width, height, categories_map):
        skin_df = pd.read_csv(os.path.join(os.path.curdir, 'data', 'dataverse_files', 'HAM10000_metadata.csv'))
        image_path = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join('data', 'dataverse_files', '*', '*.jpg'))
        }
        skin_df['path'] = skin_df['image_id'].map(image_path.get)
        # Use the path to read images.
        skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((width, height))))
        skin_df['dx'] = skin_df['dx'].replace(categories_map)

        return skin_df

    @staticmethod
    def separete_labels_images(skin_df):
        labels = skin_df['dx'].tolist()
        images = skin_df['image'].tolist()

        labels = LabelEncoder().fit_transform(labels)

        return labels, images

    @staticmethod
    def flat_images(images):
        flat_images = []

        for arr in images:
            new_arr = arr.reshape(-1)
            flat_images.append(new_arr)

        return flat_images

    @staticmethod
    def smote_image_generation(images, labels):

        oversample = SMOTE()
        images, labels = oversample.fit_resample(images, labels)

        return images, labels

    @staticmethod
    def unflat_images(images, width, height):
        dimensiones_originales = (height, width, 3)

        unflated_image_list = [np.array(arr).reshape(dimensiones_originales) for arr in images]

        return unflated_image_list