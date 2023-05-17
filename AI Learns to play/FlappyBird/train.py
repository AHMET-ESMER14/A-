import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Sequential
import numpy as np

x_train = np.load('datas.npy').reshape(-1, 10)
y_train = np.load('outs.npy').reshape(-1,)

y1 = [i for i in range(len(y_train)) if y_train[i] == 1]
y0 = [i for i in range(len(y_train)) if y_train[i] == 0]
y0 = y0[0:len(y0):len(y0) // len(y1)]


x_train = np.concatenate((x_train[y1],x_train[y0]))
y_train = np.concatenate((y_train[y1],y_train[y0]))



model = Sequential()
model.add(Dense(512, input_shape=(10,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))


model.compile( optimizer='adam',loss = tf.keras.losses.binary_crossentropy , metrics=['accuracy'])



model.fit(x_train, y_train, epochs=100, batch_size= 100,validation_split=0.1)

model.save("model.h5")
