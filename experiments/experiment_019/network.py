import tensorflow as tf

# Variable declaration
from keras.layers import BatchNormalization

data_augmentation = False
batch_size = 64
lr = 0.01
dropout = True
epochs = 35
optimizer = "Adam"  # With lr decay


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
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))  # Shape: 256x256x3
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))

        # LAYER 2
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))

        # LAYER 3
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))

        # LAYER 4
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        if dropout:
            model.add(tf.keras.layers.Dropout(0.3))

        # MLP
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(29, activation='softmax'))  # 29 Possible classes

        if self.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, decay=lr/epochs)
        elif self.optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, decay=lr/epochs)
        elif self.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=0.000001, amsgrad=True)

        model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])

        # Check model structure
        if self.verbose == 2:
            model.summary()

        return model

    def fit(self, train_data, validation_data, epochs):
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto',
                              restore_best_weights=True) #early stopping in case it overfits too much
        history = self.model.fit(x=train_data, fit_epochs=epochs, validation_data=validation_data, verbose=self.verbose, callbacks=[early])
        return history

    def test(self, test_data):
        return self.model.evaluate(test_data)
