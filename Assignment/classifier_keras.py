"""Based on: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
Changed classifier architecture to mirror the PyTorch implementation.
"""
from __future__ import print_function
try:
    import keras
    from keras import backend as K
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import backend as K

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10
input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, 5, activation='relu', input_shape=input_shape),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 5, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(600, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax'),
])


def main():
    batch_size = 64
    epochs = 100

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # Scale to [0, 1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Normalize
    mean = 0.1307
    std = 0.3081
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint('mnist_cnn.h5',
                                 verbose=1, save_best_only=True, save_weights_only=True)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.5),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[checkpoint],
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
