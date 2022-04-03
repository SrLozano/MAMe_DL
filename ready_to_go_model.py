import tensorflow as tf
from keras.models import load_model
from mame_cnn import evaluate_model, create_confusion_matrix


def load_data():
    """
    This function loads the data from dataset folder
    :return test_it_ret: Testing generator
    """

    # Create a data generator
    datagen_val_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    # Load and iterate test dataset
    test_it_ret = datagen_val_test.flow_from_directory('dataset/data_256/test/', class_mode='categorical',
                                                       batch_size=128, shuffle=False)

    return test_it_ret


if __name__ == "__main__":

    # Load model
    model = load_model('model.h5')

    # Get model description
    model.summary()

    # Load data from image generator
    test_generator = load_data()

    # Evaluate model
    evaluate_model(model, test_generator, False)

    # Get confusion matrix
    create_confusion_matrix(model, test_generator, False)
