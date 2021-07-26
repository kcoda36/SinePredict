import keras
import keras.layers as layers
from keras import Input
from keras import models
import numpy as np
import matplotlib.pyplot as plt

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs/", histogram_freq=1)

#training data
x = np.linspace(0,6.28,10000)
y = (np.sin(x) + 1) / 2
mapX = map(lambda x: [x], x)
X = list(mapX)
X = np.array(X)

# Model Creation
inp = Input(shape=(1), name="inp")
h = layers.Dense(50, activation="gelu", name="d1")(inp)
h = layers.Dense(50, activation="gelu", name="d2")(h)
h = layers.Dense(50, activation="gelu", name="d3")(h)
h = layers.Dense(50, activation="gelu", name="d4")(h)
out = layers.Dense(1, activation='sigmoid', name="out")(h)
model = models.Model(inp, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

#Train the model
model.fit(X, y, epochs=500, verbose=1, callbacks=[tensorboard_callback])

#Plot data
Y = model.predict(X)
y = (y * 2) + 1
Y = (Y * 2) + 1
plt.plot(X, Y)
plt.plot(X, y)
plt.show()

#Make prediction
print(model.predict([6.28]))

#Save Model
model.save('saved_model/my_model')