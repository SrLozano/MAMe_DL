import os
import shutil

import tensorflow as tf
import keras
import pandas as pd

'''from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt'''

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


'''# Define the NN architecture

# Two hidden layers
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(256, 256, 3)))
model.add(Dense(29, activation=tf.nn.softmax))

# Compile the NN
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Start training
history = model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)

# Evaluate model
loss = model.evaluate_generator(test_it, steps=24)

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()'''
