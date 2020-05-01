import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.initializers import glorot_uniform, RandomNormal
from keras.regularizers import l2
from keras import backend as k


def display_cost(cost):
    """
    This function is used to plot the cost values vs number of iterations.
    :param cost: Dictionary recording training and validation loss values
    :return: None
    """

    plt.plot(cost.history['loss'], label='train_loss')
    plt.plot(cost.history['val_loss'], label='val_loss')
    plt.plot(cost.history['acc'], label='train_acc')
    plt.plot(cost.history['val_acc'], label='val_acc')
    plt.ylabel('Cost/Accuracy')
    plt.xlabel('Epoch #')
    plt.title("Model Loss/Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


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
    x = Dense(units=1, activation='linear', name='OUTPUT')(x)

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
    max_iterations = 10
    mini_batch_size = 128

    # Setup the Keras CNN Model
    keras_model = model(train_x.shape[1])

    # Compile the model to configure the learning process
    keras_model.compile(loss='mean_squared_error',
                        optimizer=keras.optimizers.Adam(),
                        metrics=['mse'])

    # Train the model
    history = keras_model.fit(x=train_x, y=train_y,
                              batch_size=mini_batch_size,
                              epochs=max_iterations,
                              verbose=1,
                              validation_data=(holdout_x, holdout_y))

    # Test/evaluate the model
    score = keras_model.evaluate(x=test_x, y=test_y, verbose=0)
    print('Test loss: {}', format(score[0]))
    print('Test accuracy: {}', format(score[1] * 100))

    # Display the cost function graph
    display_cost(history)
