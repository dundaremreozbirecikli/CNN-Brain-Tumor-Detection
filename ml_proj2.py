import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import Sequence
import matplotlib.pyplot as plt
import json


X_train = np.load('X_train.npy')
X_valid = np.load('X_valid.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_valid = np.load('y_valid.npy')
y_test = np.load('y_test.npy')


print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)


#hyperparameters
epoch = 6
dropout = 0.5
lr = 0.000065
batch_size = 32

#model definiton
#filters and kernel sizes are from the article
model = Sequential()
model.add(Conv2D(filters = 128, kernel_size = (5, 5), activation="relu", input_shape=(250, 250, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 256, kernel_size = (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(rate = dropout))
model.add(Dense(units = 512, activation="relu"))
model.add(Dense(units = 256, activation="relu"))
model.add(Dense(units = 4, activation = "softmax"))

#build the model
model.build(input_shape = X_train.shape)

#define the optimizer and loss functions
model.compile(
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = lr),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

print(model.summary())
  

#a fix for the unnecessary gpu load
#TensorFlow is trying to load the full numpy array into the GPU memory
#thus gives GPU memory errors
#this is a simple fix 
#by using generators:
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


train_gen = DataGenerator(X_train, y_train, batch_size)
valid_gen = DataGenerator(X_valid, y_valid, batch_size)

#train the model
history = model.fit(train_gen, epochs = epoch, validation_data = valid_gen)

model.save('ml_proj_model.h5')

history_dict = history.history

with open("training_history_ml.json", "w") as f:
    json.dump(history_dict, f)

with open("training_params_ml.json", "w") as f:
    json.dump("Epoch:", f)
    json.dump(epoch, f)
    json.dump("Learning rate:", f)
    json.dump(lr, f)
    json.dump("Dropout:", f)
    json.dump(dropout, f)
    json.dump("Batch Size:", f)
    json.dump(batch_size, f)
