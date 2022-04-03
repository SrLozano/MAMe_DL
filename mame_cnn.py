import os
import time
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from experiments.experiment_031 import network

exp = "experiment_031"

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
    train_it_ret = datagen_train.flow_from_directory('dataset/data_256/train/', class_mode='categorical',
                                                     batch_size=batch_size)
    # Load and iterate validation dataset
    val_it_ret = datagen_val_test.flow_from_directory('dataset/data_256/val/', class_mode='categorical',
                                                      batch_size=batch_size)
    # Load and iterate test dataset
    test_it_ret = datagen_val_test.flow_from_directory('dataset/data_256/test/', class_mode='categorical',
                                                       batch_size=batch_size, shuffle=False)

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
    plt.title(f'Model {exp} accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'experiments/{exp}/MAMe_accuracy.pdf')
    plt.close()

    # Loss plot
    plt.plot(history_plot.history['loss'])
    plt.plot(history_plot.history['val_loss'])
    plt.title(f'Model {exp} loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'experiments/{exp}/MAMe_loss.pdf')


def evaluate_model(model, test_generator, experiment=True):
    """
    This functions creates interesting metrics to check model performance
    :param model:  Model to evaluate
    :param test_generator: Test generator
    :param experiment: Specifies whether ths actual run is an experiment
    :return: It saves the accuracy and loss plots
    """

    # Evaluate the model
    test_generator.reset()
    score = model.evaluate(test_generator, verbose=0)

    if experiment:
        file = open(f'experiments/{exp}/test_info.txt', "a+")
    else:
        file = open(f'model_evaluation.txt', "a+")

    file.write(f"\t - Loss: {str(score[0])} \n \t - Accuracy on test: {str(score[1])}\n")

    # Confusion Matrix (validation subset)
    test_generator.reset()
    pred = model.predict(test_generator, verbose=0)

    # Assign most probable label
    predicted_class_indices = np.argmax(pred, axis=1)

    # Get class labels
    labels = (test_generator.class_indices)
    target_names = labels.keys()

    # Plot statistics
    file.write("\n")
    file.write(classification_report(test_generator.classes, predicted_class_indices, target_names=target_names))
    file.close()


def create_confusion_matrix(model, eval_gen, experiment=True):
    """
    This function evaluates a model. It shows validation loss and accuracy, classification report and confusion matrix.
    :param model: Model to evaluate
    :param eval_gen: Evaluation generator
    :param experiment: Specifies whether ths actual run is an experiment
    """

    # Evaluate the model
    eval_gen.reset()
    score = model.evaluate(eval_gen, verbose=0)
    print('\nLoss:', score[0])
    print('Accuracy:', score[1])

    # Confusion Matrix (validation subset)
    eval_gen.reset()
    pred = model.predict(eval_gen, verbose=0)

    # Assign most probable label
    predicted_class_indices = np.argmax(pred, axis=1)

    # Get class labels
    labels = eval_gen.class_indices
    target_names = labels.keys()

    # Plot statistics
    print(classification_report(eval_gen.classes, predicted_class_indices, target_names=target_names))

    cf_matrix = confusion_matrix(np.array(eval_gen.classes), predicted_class_indices)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cf_matrix, annot=False, cmap='Blues', cbar=True, square=False,
                          xticklabels=target_names, yticklabels=target_names)
    fig = heatmap.get_figure()

    if experiment:
        fig.savefig(f'experiments/{exp}/MAMe_confusion_matrix.pdf')
    else:
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
    network = network.CNN(learning_rate=learning_rate, verbose=0, fit_optimizer=optimizer,
                          loss='categorical_crossentropy')

    if verbose_level == 2:
        network.model.summary()  # Check model structure

    # Measure elapsed time
    start = time.time()

    # Fit network
    history = network.fit(train_it, val_it, fit_epochs=epochs)

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
    create_confusion_matrix(network.model, test_it)
