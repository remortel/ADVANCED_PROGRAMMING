import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Prepare a dataset.
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Prepare our layers, loss, and optimizer.
# Play with the number of layers and the number of nodes per layer!

inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(32, activation="relu", name="dense_1")(inputs)
x = layers.Dropout(0.1)(x)
x = layers.Dense(32, activation="relu", name="dense_2")(x)
x = layers.Dropout(0.1)(x)
#x = layers.Dense(64, activation="relu", name="dense_3")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# your additional model hyperparameters are given below
batch_size = 128
lr = 1e-3
mom = 0.9
epochs = 140

print("start fitting")
# Try and play with the batch size, learning rate and optimizers
# You will see remarkable changes in results!
model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=lr, momentum=mom, nesterov=True), metrics=["CategoricalAccuracy", "Precision"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
