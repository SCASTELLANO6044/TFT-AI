import os

epochs = 1
img_width = 100
img_height = 75
model_name = "my_model.keras"

model_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'model')
media_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'media')
metadata_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files',
                             'HAM10000_metadata.csv')
images_folder_part1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files',
                                   'HAM10000_images_part_1')
images_folder_part2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dataverse_files',
                                   'HAM10000_images_part_2')
log_path_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'log.log')

categories_dict = {0: 'Enfermedad de Bowen.',
                   1: 'Carcinoma de células basales.',
                   2: 'Dermatofibroma.',
                   3: 'Lesión Vascular.',
                   4: 'Lunar común.',
                   5: 'Queratosis benigna.',
                   6: 'Melanoma.'}
categories_map = {'akiec': 0, 'bcc': 1, 'df': 2, 'vasc': 3, 'nv': 4, 'bkl': 5, 'mel': 6}

accuracy_training_histogram_file = os.path.join(media_folder, 'accuracy_training_history.png')
loss_training_histogram_file = os.path.join(media_folder, 'loss_training_history.png')
confusion_matrix_file = os.path.join(media_folder, 'confusion_matrix.png')
icon_logo_ulpgc = os.path.join(media_folder, 'logo_ulpgc_vertical_acronimo_mancheta_azul.ico')
image_logo_ulpgc = os.path.join(media_folder, 'ulpgc-logo.png')
