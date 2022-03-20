import os
import shutil

import tensorflow as tf
import keras
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def sort_dataset_folder(execute_image_sorting_bool, metadata_info):
    """
    This function sorts the dataset folder so that it can be read using generators.Thus, the folder structure is:
        data_256
        ├── test
        │   ├── Bronze
        │   │   └── 1914.572.jpg
        │   └── Ceramic
        │       └── 1914.593.jpg
        ├── train
        │   ├── Bronze
        │   │   └── 1914.132.jpg
        │   └── Ceramic
        │       └── 1914.456.jpg
        ├── val
        │   ├── ├── Bronze
        │   │   └── 1914.383.jpg
        │   └── Ceramic
        │       └── 1914.537.jpg
    :param metadata_info: dataframe with metadata from the dataset
    :param execute_image_sorting_bool: boolean to control whether the dataset folder gets sorted
    :return:
    """

    if execute_image_sorting_bool:

        subset_names = list(metadata_info['Subset'].unique())

        for i in subset_names:
            os.makedirs(os.path.join('dataset/data_256', i))

        for c in subset_names:
            for i in list(metadata_info[metadata_info['Subset'] == c]['Image file']):

                # Create path to the image
                get_image = os.path.join('dataset/data_256', i)

                # If image has not already exist in the new folder create one
                if not os.path.exists('dataset/data_256/' + c + i):
                    # Move the image
                    move_image_to_cat = shutil.move(get_image, 'dataset/data_256/' + c)

        classes_names = list(metadata_info['Medium'].unique())

        for subset in ['train', 'val', 'test']:

            for i in classes_names:
                os.makedirs(os.path.join(f'dataset/data_256/{subset}/', i))

            for c in classes_names:
                aux_df = metadata_info.loc[(metadata_info['Medium'] == c) & (metadata_info['Subset'] == subset)]
                for index, row in aux_df.iterrows():

                    # Create path to the image
                    get_image = os.path.join(f'dataset/data_256/{subset}', row['Image file'])

                    # If image has not already exist in the new folder create one
                    if not os.path.exists(f'dataset/data_256/{subset}/' + c + '/' + row['Image file']):
                        # Move the image
                        move_image_to_cat = shutil.move(get_image,
                                                        f'dataset/data_256/{subset}/' + c + '/' + row['Image '
                                                                                                      'file'])


def load_dataset():
    """
    This function loads the dataset from dataset folder
    :return train_it_ret: Training generator
    :return val_it_ret: Validation generator
    :return test_it_ret: Testing generator
    """

    # Create a data generator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # Load and iterate training dataset
    train_it_ret = datagen.flow_from_directory('dataset/data_256/train/', class_mode='categorical', batch_size=32)
    # Load and iterate validation dataset
    val_it_ret = datagen.flow_from_directory('dataset/data_256/val/', class_mode='categorical', batch_size=32)
    # Load and iterate test dataset
    test_it_ret = datagen.flow_from_directory('dataset/data_256/test/', class_mode='categorical', batch_size=32)

    return train_it_ret, val_it_ret, test_it_ret


def create_plots(history_plot):
    """
    This functions creates the plots accuracy and loss evolution in training and validation
    :param history_plot: Record of training loss values and metrics values at successive epochs
    :return: It saves the accuracy and loss plots
    """
    # Accuracy plot
    plt.plot(history_plot.history['accuracy'])
    plt.plot(history_plot.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('MAMe_accuracy.pdf')
    plt.close()

    # Loss plot
    plt.plot(history_plot.history['loss'])
    plt.plot(history_plot.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('MAMe_loss.pdf')


def create_confusion_matrix(model_cf, test_generator, metadata_info):
    """
    This functions creates a confusion matrix for the test set
    :param model_cf: model to evaluate
    :param test_generator: image generator corresponding to test set
    :param metadata_info: metadata of the dataset
    :return:
    """
    # Compute probabilities
    y_pred_aux = model_cf.predict(test_generator)

    # Assign most probable label
    y_pred = np.argmax(y_pred_aux, axis=1)

    # target_names = sorted(list(metadata_info['Medium'].unique()))
    cf = confusion_matrix(np.argmax(test_generator, axis=1), y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cf, annot=True, cmap="Blues")
    plt.savefig('MAMe_confusion_matrix.pdf')


if __name__ == "__main__":

    print("Hello")

    # Variable declaration
    execute_image_sorting = False
    verbose_level = 1  # 0: Silent, 1: Minimum detail, 2: Maximum detail

    matplotlib.use('Agg')  # Select the backend used for rendering and GUI integration.
    if verbose_level == 2:
        print('Using Keras version', keras.__version__)  # Display Keras version

    metadata = pd.read_csv("dataset/MAMe_metadata/MAMe_dataset.csv")

    # Sort dataset so that it can be read with image generators
    sort_dataset_folder(execute_image_sorting, metadata)

    # Load dataset
    train_it, val_it, test_it = load_dataset()

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

    if verbose_level == 2:
        model.summary()  # Check model structure

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit network
    history = model.fit(
        x=train_it,
        epochs=3,
        validation_data=val_it,
        verbose=0)

    # Test model
    test_lost, test_acc = model.evaluate(test_it)
    if verbose_level >= 1:
        print("Test Accuracy:", test_acc)

    # Create plots
    create_plots(history)

    print("Create files")
    '''
    # Create txt file with important information about network performance
    id_experiment = "000_experiment"
    f = open(f"{id_experiment}.txt", "w+")
    f.write(f"Experiment {id_experiment} \n Test Accuracy: {test_acc}")

    target_names = sorted(list(metadata['Medium'].unique()))
    y_pred_aux = model.predict(test_it)
    y_pred = np.argmax(y_pred_aux, axis=1)

    f.write(classification_report(np.argmax(test_it, axis=1), y_pred, target_names=target_names))
    f.close()'''

    print("Confusion Matrix")

    '''# Plot confusion  matrix
    create_confusion_matrix(model, test_it, metadata)'''
