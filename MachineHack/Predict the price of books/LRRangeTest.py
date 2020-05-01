import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.initializers import RandomNormal
from learningratefinder import LearningRateFinder
from keras import backend as k


def root_mean_squared_error(y_true, y_pred):
    """
    This function is used to calculate the root mean squared error.
    :param y_true: True prices of books
    :param y_pred: Predicted prices of books
    :return: RMSE value
    """

    return k.sqrt(k.mean(k.square(y_pred - y_true)))


def model(num_of_features):
    """
    This function is used to train the CNN model and get optimal model parameters.
    :param num_of_features: Number of features in input data
    :return: Keras CNN Model
    """

    # Input Layer
    x_input = Input(shape=(num_of_features, ), name='INPUT')

    # Fully-connected Layer 1
    x = Dense(units=512, activation='relu', name='FC-1', kernel_initializer=RandomNormal(seed=1))(x_input)
    x = BatchNormalization(name='BN_FC-1')(x)
    x = Dropout(rate=0.5, name='DROPOUT_FC-1')(x)

    # Fully-connected Layer 2
    x = Dense(units=128, activation='relu', name='FC-2', kernel_initializer=RandomNormal(seed=1))(x)
    x = BatchNormalization(name='BN_FC-2')(x)
    x = Dropout(rate=0.5, name='DROPOUT_FC-2')(x)

    # Output Layer
    x = Dense(units=1, activation='linear', name='OUTPUT', kernel_initializer=RandomNormal(seed=1))(x)

    # Create Keras Model instance
    nn_model = Model(inputs=x_input, outputs=x, name='BooksPricePredictor')
    return nn_model


if __name__ == "__main__":

    # Read the training, holdout and test datasets from processed file
    processed_dataset = np.load('DataSet/processed_dataset.npz', allow_pickle=True)
    train_x = processed_dataset['X_train']
    train_y = processed_dataset['Y_train']
    holdout_x = processed_dataset['X_val']
    holdout_y = processed_dataset['Y_val']
    test_x = processed_dataset['X_test']
    test_y = processed_dataset['Y_test']
    predict_x = processed_dataset['X_predict']

    print("----------------------- Training Dataset --------------------------")
    print("train_x shape: {}".format(train_x.shape))
    print("train_y shape: {}".format(train_y.shape))

    print("\n----------------------- Holdout Dataset --------------------------")
    print("holdout_x shape: {}".format(holdout_x.shape))
    print("holdout_y shape: {}".format(holdout_y.shape))

    print("\n----------------------- Test Dataset --------------------------")
    print("test_x shape: {}".format(test_x.shape))
    print("test_y shape: {}".format(test_y.shape))

    print("\n----------------------- Predict Dataset --------------------------")
    print("predict_x shape: {}".format(predict_x.shape))

    # Define the model hyperparameters
    mini_batch_size = 128

    # Setup the Keras CNN Model
    keras_model = model(train_x.shape[1])

    # Compile the model to configure the learning process
    keras_model.compile(loss=root_mean_squared_error,
                        optimizer=keras.optimizers.Adam(),
                        metrics=['mse'])

    # Learning Rate Finder
    lrf = LearningRateFinder(keras_model)
    lrf.find((train_x, train_y),
             startLR=1e-10, endLR=1e+1,
             stepsPerEpoch=np.ceil((len(train_x) / float(mini_batch_size))),
             batchSize=mini_batch_size)
    lrf.plot_loss()
    plt.grid()
    plt.show()
