import tensorflow as tf
import numpy as np
from keras.utils import Sequence
import json

X_train = np.load('X_train.npy')
X_valid = np.load('X_valid.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_valid = np.load('y_valid.npy')
y_test = np.load('y_test.npy')


model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(250, 250, 3),
    classes=4)

lr = 0.000075
batch_size = 32
epoch = 15

#a fix for the unnecessary gpu load
#TensorFlow is trying to load the full numpy array into the GPU memory
#thus gives GPU memory errors
#this is a simple fix that I have found from stackoverflow
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

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#define the optimizer and loss functions
model.compile(
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = lr),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)
history = model.fit(train_gen, epochs=epoch, batch_size=batch_size, validation_data=valid_gen)

model.save('ml_proj_model.h5')

history_dict = history.history

with open("training_history_ml_2.json", "w") as f:
    json.dump(history_dict, f)

with open("training_params_ml_2.json", "w") as f:
    json.dump("Epoch:", f)
    json.dump(epoch, f)
    json.dump("Learning rate:", f)
    json.dump(lr, f)
    json.dump("Batch Size:", f)
    json.dump(batch_size, f)