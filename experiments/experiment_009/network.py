import tensorflow as tf


# Variable declaration
from keras.layers import BatchNormalization

data_augmentation = False
batch_size = 128 #now with 128
lr = 0.01
dropout = True
epochs = 24
optimizer = "SGD" #with lr decay
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
        #LAYER 1
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))  # Shape: 256x256x3
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))
        #LAYER 2
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))  # Shape: 256x256x3
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))
        #LAYER 3
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(256, 256, 3)))  # Shape: 256x256x3
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))
        #LAYER 4
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(256, 256, 3)))  # Shape: 256x256x3
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(29, activation='softmax'))  # 29 Possible classes

        if self.verbose == 2:
            model.summary()  # Check model structure

        if self.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9,decay = lr / epochs) #we add the lr decay
            # TODO: add momentum y decay a los parameteros, en funcion de si llamamos al SGD o al RMS prop
        elif self.optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, decay=lr / epochs)

        model.compile(optimizer=optimizer,
                      loss=self.loss, metrics=['accuracy'])

        return model

    def fit(self, train_data, validation_data, epochs):
        history = self.model.fit(x=train_data, fit_epochs=epochs, validation_data=validation_data, verbose=self.verbose)
        self.model = history
        return history

    def test(self, test_data):
        return self.model.evaluate(test_data)
