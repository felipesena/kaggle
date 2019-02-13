import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

MODEL_JSON_FILE = 'model.json'
EPOCHS = 30


# define the larger model
def create_model(num_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def save_model(model):
    model_json = model.to_json()
    with open(MODEL_JSON_FILE, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    print('Model was save succesfully')


def load_model():
    json_file = open(MODEL_JSON_FILE, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')

    print('Model was load succesfully')

    return loaded_model


def normalize(input):
    input = input.reshape(input.shape[0], 1, 28, 28).astype('float32')
    input = input / 255.0
    return input


def get_model():
    if os.path.isfile(MODEL_JSON_FILE):
        model = load_model()
    else:
        mnist_train = np.genfromtxt('train.csv', delimiter=',', skip_header=1)

        y_train = mnist_train[:, 0]
        X_train = mnist_train[:, 1:]

        seed = 7
        np.random.seed(seed)

        X_train = normalize(X_train)

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)

        model = create_model(y_train.shape[1])

        # Fit the model
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=200)

        save_model(model)

    return model


if __name__ == '__main__':

    X_test = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

    model = get_model()

    X_test = normalize(X_test)

    results = model.predict(X_test, verbose=1)

    results = np.argmax(results, axis=1)

    predict_values = pd.Series(results, name='Label')

    submission = pd.concat([pd.Series(range(1, X_test.shape[0] + 1), name="ImageId"), predict_values], axis=1)

    submission.to_csv('submission.csv', index=False)
