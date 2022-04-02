import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import GlobalMaxPooling2D

# Variable declaration
data_augmentation = False
batch_size = 128
batch_normalization = False
lr = 0.001
dropout = True
dropout_perc = 0.1
epochs = 40
optimizer = "Adam"
complexity = 'low'


class CNN:
    def __init__(self, learning_rate, verbose, optimizer, loss):
        # Add the needed parameters
        self.learning_rate = learning_rate
        self.verbose = verbose  # 0: silent, 1: minimum detail, 2: full explanation
        self.optimizer = optimizer
        self.loss = loss
        self.model = self.generate_model()

    def generate_model(self):
        model = tf.keras.models.Sequential()

        # LAYER 1
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal",
                                         input_shape=(256, 256, 3)))  # Shape: 256x256x3
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if complexity == 'high':
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                             input_shape=(256, 256, 3)))  # Shape: 256x256x3
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout_perc))

        # LAYER 2
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if complexity == 'high':
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout_perc))

        # LAYER 3
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if complexity == 'high':
            model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout_perc))

        # LAYER 4
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if complexity == 'high':
            model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout_perc))

        # Global average pooling
        model.add(GlobalMaxPooling2D())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(29, activation='softmax'))  # 29 Possible classes

        if self.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, decay=lr / epochs)
        elif self.optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, decay=lr / epochs)
        elif self.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-8, amsgrad=True)

        model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def fit(self, train_data, validation_data, epochs):
        history = self.model.fit(x=train_data, epochs=epochs, validation_data=validation_data, verbose=self.verbose)
        return history

    def test(self, test_data):
        return self.model.evaluate(test_data)
