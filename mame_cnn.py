import os
import shutil

import tensorflow as tf
import keras
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


'''def sort_dataset_folder(execute_image_sorting):
    """
    This function sorts the dataset folder so that it can be read using generators.Thus, the folder structure is:

    :param execute_image_sorting: boolean to control whether the dataset folder gets sorted
    :return:
    """
    # True if we want the script to sort all the images

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
                        move_image_to_cat = shutil.move(get_image,
                                                        f'dataset/data_256/{subset}/' + c + '/' + row['Image '
                                                                                                      'file'])


if __name__ == "main":
    print("Hello")
'''

matplotlib.use('Agg')  # Select the backend used for rendering and GUI integration.

print('Using Keras version', keras.__version__)

metadata = pd.read_csv("dataset/MAMe_metadata/MAMe_dataset.csv")

print(metadata)

# Create a data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Load and iterate training dataset
train_it = datagen.flow_from_directory('dataset/data_256/train/', class_mode='categorical', batch_size=32)
# Load and iterate validation dataset
val_it = datagen.flow_from_directory('dataset/data_256/val/', class_mode='categorical', batch_size=32)
# Load and iterate test dataset
test_it = datagen.flow_from_directory('dataset/data_256/test/', class_mode='categorical', batch_size=32)

# Define model structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),  # Input Shape: 256x256x3
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')  # 29 Possible classes
])

# Check model structure
model.summary()

# Compile model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Fit Network
history = model.fit(
    x=train_it,
    epochs=10,
    validation_data=val_it,
    verbose=2)

# Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('MAMe_accuracy.pdf')
plt.close()

# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('MAMe_loss.pdf')

# Test model
test_lost, test_acc = model.evaluate(test_it)
print("Test Accuracy:", test_acc)
