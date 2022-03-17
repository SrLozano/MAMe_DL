import os
import shutil

import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt

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
history = model.fit_generator(
             train_it,
             epochs=50,
             validation_data=val_it,
             verbose=2)

# Plot and show results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1, 1)

plt.plot(epochs, acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r--')
plt.plot(epochs, val_loss,  'b')
plt.title('Training and validation loss')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()

# Test model
test_lost, test_acc = model.evaluate_generator(test_it)
print("Test Accuracy:", test_acc)


'''# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()'''
