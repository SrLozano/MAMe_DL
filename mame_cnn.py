import os
import shutil

import tensorflow as tf
import keras
import pandas as pd

print('Using Keras version', keras.__version__)

metadata = pd.read_csv("dataset/MAMe_metadata/MAMe_dataset.csv")

print(metadata)

# True if we want the script to sort all the images
execute_image_sorting = False

if execute_image_sorting:

    subset_names = list(metadata['Subset'].unique())

    for i in subset_names:
        os.makedirs(os.path.join('dataset/data_256', i))

    for c in subset_names:
        for i in list(metadata[metadata['Subset'] == c]['Image file']):

            # Create path to the image
            get_image = os.path.join('dataset/data_256', i)

            # If image has not already exist in the new folder create one
            if not os.path.exists('dataset/data_256/' + c + i):
                # Move the image
                move_image_to_cat = shutil.move(get_image, 'dataset/data_256/' + c)

    classes_names = list(metadata['Medium'].unique())

    for subset in ['train', 'val', 'test']:

        for i in classes_names:
            os.makedirs(os.path.join(f'dataset/data_256/{subset}/', i))

        for c in classes_names:
            aux_df = metadata.loc[(metadata['Medium'] == c) & (metadata['Subset'] == subset)]
            for index, row in aux_df.iterrows():

                # Create path to the image
                get_image = os.path.join(f'dataset/data_256/{subset}', row['Image file'])

                # If image has not already exist in the new folder create one
                if not os.path.exists(f'dataset/data_256/{subset}/' + c + '/' + row['Image file']):
                    # Move the image
                    move_image_to_cat = shutil.move(get_image, f'dataset/data_256/{subset}/' + c + '/' + row['Image '
                                                                                                             'file'])

# Create a data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('dataset/data_256/train/', class_mode='categorical', batch_size=32)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('dataset/data_256/val/', class_mode='categorical', batch_size=32)
# load and iterate test dataset
test_it = datagen.flow_from_directory('dataset/data_256/test/', class_mode='categorical', batch_size=32)

'''path = os.path.join(os.getcwd(), "dataset/")
'''
'''os.mkdir('tempDir')

for _, row in metadata.iterrows():
    file = row['Image file']
    label = row['Subset']
    os.replace(os.path.join(os.getcwd(), f'data_256/{file}'), os.path.join(os.getcwd(), f'data_256/{label}/{file}'))'''

'''tf.keras.utils.image_dataset_from_directory(
    directory=path,
    batch_size=32,
    image_size=(256, 256),
    labels=tuple(metadata['Medium'])
)'''
