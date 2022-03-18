import os
import shutil

import tensorflow as tf
import keras
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def sort_dataset_folder(execute_image_sorting_bool):
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
    :param execute_image_sorting_bool: boolean to control whether the dataset folder gets sorted
    :return:
    """

    metadata = pd.read_csv("dataset/MAMe_metadata/MAMe_dataset.csv")

    if execute_image_sorting_bool:

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


if __name__ == "__main__":

    print("Hello")

    # Variable declaration
    execute_image_sorting = False
    verbose_level = 2  # 0: Silent, 1: Minimum detail, 2: Maximum detail

    matplotlib.use('Agg')  # Select the backend used for rendering and GUI integration.
    if verbose_level == 2:
        print('Using Keras version', keras.__version__)  # Display Keras version

    # Sort dataset so that it can be read with image generators
    sort_dataset_folder(execute_image_sorting)

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

    # Fit Network
    history = model.fit(
        x=train_it,
        epochs=10,
        validation_data=val_it,
        verbose=2)

    # Test model
    test_lost, test_acc = model.evaluate(test_it)
    if verbose_level >= 1:
        print("Test Accuracy:", test_acc)

    # Create plots
    create_plots(history)
