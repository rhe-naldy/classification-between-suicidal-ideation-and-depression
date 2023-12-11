import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
#from keras.callbacks import ModelCheckpoint

#from sklearn.neural_network import MLPClassifier as MLP
#from sklearn import metrics
#from sklearn.metrics import accuracy_score, f1_score , recall_score, precision_score

from keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, MaxPooling1D, Dropout, Conv1D, Input, LSTM, SpatialDropout1D, Bidirectional

import matplotlib.pyplot as plt

train_labels = pd.read_csv("train_labels.csv", header=None)[0]
test_labels = pd.read_csv("test_labels.csv", header=None)[0]

train_labels = np.asarray(train_labels).astype('float32').reshape((-1, 1))
test_labels = np.asarray(test_labels).astype('float32').reshape((-1, 1))

train_features = pd.read_csv("bert-training-features.csv", header=None)
test_features = pd.read_csv("bert-testing-features.csv", header=None)

epochs = 80
batch_size = 32

filters = 3
kernel = 2

# Recurrent Neural Network

rnn = Sequential()
rnn.add(Input(shape=(768, 1)))
rnn.add(layers.GRU(3, return_sequences=True, activation='relu', kernel_initializer='he_uniform'))
rnn.add(Flatten())
rnn.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
rnn.add(Dense(1, activation='sigmoid'))

rnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=tf.keras.metrics.BinaryAccuracy())

rnn.summary()

rnn.fit(train_features, train_labels, epochs=epochs, batch_size = batch_size)

pred = rnn.predict(test_features, batch_size=batch_size)

m1 = tf.keras.metrics.BinaryAccuracy()
m1.update_state(test_labels, pred)
print("Accuracy: " + str(m1.result().numpy()))

m2 = tf.keras.metrics.Precision()
m2.update_state(test_labels, pred)
print("Precision: " + str(m2.result().numpy()))

m3 = tf.keras.metrics.Recall()
m3.update_state(test_labels, pred)
print("Recall: " + str(m3.result().numpy()))

f1 = 2 * (m3.result().numpy() * m2.result().numpy()) / (m3.result().numpy() + m2.result().numpy())
print("f1: " + str(f1))

m5 = tf.keras.metrics.AUC()
m5.update_state(test_labels, pred)
print("AUC: " + str(m5.result().numpy()))
