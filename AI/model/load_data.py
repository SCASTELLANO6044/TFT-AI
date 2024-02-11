import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import time
import shutil
import torchvision
from torchvision import transforms
import torch.utils.data as data

def process_data_into_pandas_df():
    start_time = time.time()

    skin_df = pd.read_csv(os.path.join('.', 'data', 'dataverse_files', 'HAM10000_metadata.csv'))

    image_path = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join('data', 'dataverse_files', '*', '*.jpg'))
    }

    skin_df['path'] = skin_df['image_id'].map(image_path.get)
    #Use the path to read images.
    skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Tiempo transcurrido:", elapsed_time, "segundos")

def process_data_into_keras():
    datagen = ImageDataGenerator()

    # define training directory that contains subfolders
    train_dir = os.path.join(os.getcwd(), 'data', 'reorganized')
    # USe flow_from_directory
    train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                                   class_mode='categorical',
                                                   batch_size=16,  # 16 images at a time
                                                   target_size=(32, 32))  # Resize images

def process_data_into_pytorch():

    # Define root directory with subdirectories
    train_dir = os.path.join(os.getcwd(), 'data', 'reorganized')

    # If you want to apply ransforms
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts your input image to PyTorch tensor.
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # With transforms
    # train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    # Without transforms
    train_data_torch = torchvision.datasets.ImageFolder(root=train_dir)
    # train_data_loader_torch = data.DataLoader(train_data_torch, batch_size=len(train_data_torch))

    print("Number of train samples: ", len(train_data_torch))
    print("Detected Classes are: ", train_data_torch.class_to_idx)  # classes are detected by folder structure

    labels = np.array(train_data_torch.targets)
    (unique, counts) = np.unique(labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)
