import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import gaussian_filter
import tensorflow_addons as tfa


class_labels = {'glioma':0, 'meningioma':1, 'notumor':2, 'pituitary':3}

#Train iamges seperation
path = "archive-4/Training"
train_data = []
train_label = []
for label in os.listdir(path):
    path1 = os.path.join(path,label)
    if(label == ".DS_Store"):
        continue
    for file in os.listdir(path1):
        image = Image.open(path1+"/"+file)
        image = image.convert("RGB")
        image = image.resize((250,250))
        image = np.array(image)
        train_data.append(image)
        train_label.append(class_labels[label])


#Test images seperation
path = "archive-4/Testing"
X_test = []
y_test = []
for label in os.listdir(path):
    path1 = os.path.join(path,label)
    if(label == ".DS_Store"):
        continue
    for file in os.listdir(path1):
        image = Image.open(path1+"/"+file)
        image = image.convert("RGB")
        image = image.resize((250,250))
        image = np.array(image)
        X_test.append(image)
        y_test.append(class_labels[label])


class_labels_list = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']


#shuffle the train set
train_x, train_y = shuffle(train_data, train_label, random_state=41)

X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, shuffle=False, random_state=41)

"""
plt.figure(figsize=(7,20))
i=1
for img,label in zip(X_train[:10],y_train[:10]):
    plt.subplot(5,2,i)
    plt.imshow(img)
    plt.title(f"Label:{class_labels_list[label]}", fontsize = 18)
    i+=1

plt.savefig("ml_fig1.png", dpi = 600)

"""


y_train = np.array(train_y)
X_train = np.array(train_x)

X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

X_test = np.array(X_test)
y_test = np.array(y_test)


print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)

x_train = tf.convert_to_tensor(X_train)
x_valid = tf.convert_to_tensor(X_valid)
x_test = tf.convert_to_tensor(X_test)

image_ops = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

for i in range(len(X_train)):
    X_train[i] = image_ops(x_train[i])

for i in range(len(X_valid)):
    X_valid[i] = image_ops(x_valid[i])

for i in range(len(X_test)):
    X_test[i] = image_ops(x_test[i])


train_x = []
train_y = []

#we get each image 
#flip, and rotate by 90 and 270 degrees
#and enlarge our dataset (1 image -> 4 images)
#and add the corresponding labels
for i in range(len(X_train)):
    train_x.append(X_train[i])
    train_x.append(tf.image.flip_left_right(X_train[i]))
    train_x.append(tfa.image.rotate(images=X_train[i], angles=tf.constant((np.pi)/2)))
    train_x.append(tfa.image.rotate(images=X_train[i], angles=tf.constant((np.pi * 3)/2)))
    train_y.append(y_train[i])
    train_y.append(y_train[i])
    train_y.append(y_train[i])
    train_y.append(y_train[i])    

y_train = np.array(train_y)
X_train = np.array(train_x)

X_valid = np.array(X_valid)
X_test = np.array(X_test)


np.save('X_train.npy', X_train)
np.save('X_valid.npy', X_valid)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_valid.npy', y_valid)
np.save('y_test.npy', y_test)
