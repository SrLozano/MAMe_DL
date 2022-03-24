import os
import shutil

import tensorflow as tf
import keras
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

import time

from experiments.experiment_016 import network
exp="experiment_016"

data_augmentation = network.data_augmentation
batch_size = network.batch_size
learning_rate = network.lr
epochs = network.epochs
optimizer = network.optimizer

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
                os.makedirs(os.path.join(f'dataset/data_256/{subset}/', i.strip()))

            for c in classes_names:
                aux_df = metadata_info.loc[(metadata_info['Medium'] == c) & (metadata_info['Subset'] == subset)]
                for index, row in aux_df.iterrows():

                    # Create path to the image
                    get_image = os.path.join(f'dataset/data_256/{subset}', row['Image file'])

                    # If image does not already exist in the new folder create one
                    if not os.path.exists(f'dataset/data_256/{subset}/' + c.strip() + '/' + row['Image file']):
                        # Move the image
                        move_image_to_cat = shutil.move(get_image,
                                                        f'dataset/data_256/{subset}/' + c.strip() + '/' + row['Image '
                                                                                                      'file'])


def load_dataset(data_augmentation=False):
    """
    This function loads the dataset from dataset folder
    :parameter data_augmentation: Boolean indicating whether to include or not data augmentation
    :return train_it_ret: Training generator
    :return val_it_ret: Validation generator
    :return test_it_ret: Testing generator
    """

    # Create a data generator
    if data_augmentation:
        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
                                                                        rotation_range=20,
                                                                        width_shift_range=0.1,
                                                                        height_shift_range=0.1,
                                                                        zoom_range=0.2,
                                                                        horizontal_flip=True)
    else:
        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    datagen_val_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    # Load and iterate training dataset
    train_it_ret = datagen_train.flow_from_directory('dataset/data_256/train/', class_mode='categorical', batch_size=batch_size)
    # Load and iterate validation dataset
    val_it_ret = datagen_val_test.flow_from_directory('dataset/data_256/val/', class_mode='categorical', batch_size=batch_size)
    # Load and iterate test dataset
    test_it_ret = datagen_val_test.flow_from_directory('dataset/data_256/test/', class_mode='categorical', batch_size=batch_size)

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
    plt.title(f'model {exp} accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'experiments/{exp}/MAMe_accuracy.pdf')
    plt.close()

    # Loss plot
    plt.plot(history_plot.history['loss'])
    plt.plot(history_plot.history['val_loss'])
    plt.title(f'model {exp} loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'experiments/{exp}/MAMe_loss.pdf')


def evaluate_model(model, test_generator):
    """
    This functions creates interesting metrics to check model performance
    :param model:  Model to evaluate
    :param test_generator: Test generator
    :return: It saves the accuracy and loss plots
    """

    # Evaluate the model
    test_generator.reset()
    score = model.evaluate(test_generator, verbose=0)

    f = open(f'experiments/{exp}/test_info.txt', "a+")
    f.write(f"\t - Loss: {str(score[0])} \n \t - Accuracy on test: {str(score[1])}\n")


    # Confusion Matrix (validation subset)
    test_generator.reset()
    pred = model.predict(test_generator, verbose=0)

    # Assign most probable label
    predicted_class_indices = np.argmax(pred, axis=1)

    # Get class labels
    labels = (test_generator.class_indices)
    target_names = labels.keys()

    # Plot statistics
    f.write("\n")
    f.write(classification_report(test_generator.classes, predicted_class_indices, target_names=target_names))
    f.close()

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
    heatmap = sns.heatmap(cf, annot=True, cmap="Blues")
    fig = heatmap.get_figure()
    fig.savefig('MAMe_confusion_matrix.pdf')


if __name__ == "__main__":

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
    train_it, val_it, test_it = load_dataset(data_augmentation)

    # Define model structure
    network = network.CNN(learning_rate=learning_rate, verbose=0, optimizer=optimizer, loss='categorical_crossentropy')

    if verbose_level == 2:
        network.model.summary()  # Check model structure

    # Measure elapsed time
    start = time.time()

    # Fit network
    history = network.fit(train_it, val_it, epochs=epochs)

    # Measure elapsed time
    end = time.time()
    time_taken = (((end - start)/60)/60)

    # Create plots
    create_plots(history)

    # Create txt file with important information about network performance
    f = open(f'experiments/{exp}/test_info.txt', "w+")
    f.write(f"Experiment {exp} \n \t - Time elapsed: {time_taken} \n")
    f.close()

    # Evaluate model
    evaluate_model(network.model, test_it)

    # Plot confusion  matrix
    # create_confusion_matrix(network.model, test_it, metadata) # TODO: esto no deberíamos hacerlo antes con validation? dijo que test solo al final
