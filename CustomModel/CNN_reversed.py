from keras.models import Sequential
import keras_genomics
from keras.layers import MaxPooling1D, Dropout


def cnn_reversed(window) :
    """
        Create a convolutional model with 3 convolutional layers before a final 
        dense a layer with one node used to make the final prediction.
        
        ..notes: the precision of the prediction does not depend strongly with the architecture.
    """
    num_classes = 1
    
    model = Sequential()
    model.add(keras_genomics.layers.RevCompConv1D(
                filters=64, kernel_size=12,
                input_shape=(window, 4), activation="relu",
                padding="same"))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    model.add(Dropout(0.2))
    
    model.add(keras_genomics.layers.RevCompConv1D(
                filters=16, kernel_size=8,
                activation="relu",
                padding="same"))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    model.add(Dropout(0.2))
    
    model.add(keras_genomics.layers.RevCompConv1D(
                filters=8, kernel_size=4,
                activation="relu",
                padding="same"))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(keras_genomics.layers.normalization.RevCompConv1DBatchNorm())
    model.add(Dropout(0.2))
    
    model.add(keras_genomics.layers.core.DenseAfterRevcompConv1D(
                units=64, activation="relu"))
    model.add(keras_genomics.layers.core.Dense(
                units=num_classes))

    return model 