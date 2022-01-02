import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, regularizers, backend

import librosa



librosa.feature.mfcc()
numberOfCoeficients = 13
frameLength = 0.02
frameStride = 0.02
filterNumber = 32
FFTLength = 256
normalizationWindowSize = 101
lowFrequency = 300

model = models.Sequential([

    # Reshape input
    layers.InputLayer(650),
    layers.Reshape(target_shape=13),

    # Convolutional layer
    layers.Conv1D(kernel_size=3,neurons=8)
    layers.Dropout(0.25),

    layers.Conv1D(kernel_size=3, neurons=16)
    layers.Dropout(0.25),

    # Flatten
    layers.Flatten(),

    # # Classifier
    # layers.Dense(32, activation='relu'),
    # layers.Dropout(0.5),
     layers.Dense(2, activation=tf.nn.softmax)
])