import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns

#for each model, load the trained weights
with open("training_history_ml.json", "r") as f:
    history = json.load(f)

model = tf.keras.models.load_model("ml_proj_model.h5")


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

"""
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history['accuracy'], label='Accuracy')
plt.plot(history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
"""

#evaluate the model
test_loss, test_acc = model.evaluate(X_test,  y_test)

valid_loss, valid_acc = model.evaluate(X_valid, y_valid)

print("Test Accuracy:",test_acc)
print("Test Loss:", test_loss)

print("Validation Accuracy:",valid_acc)
print("Validation Loss:", valid_loss)

test_predict = model.predict(X_test)

predicted_probabilities_test = tf.nn.softmax(test_predict, axis=-1)

test_result = tf.argmax(predicted_probabilities_test, axis=1)



valid_predict = model.predict(X_valid)

predicted_probabilities_valid = tf.nn.softmax(valid_predict, axis=-1)

valid_result = tf.argmax(predicted_probabilities_valid, axis=1)


conf_matrix = confusion_matrix(y_test, test_result)

class_labels_list = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

plt.figure(figsize=(30,30))
sns.set(font_scale=5.0)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=2.8,
            annot_kws={"size": 75}, cbar_kws={'shrink': 1.0}, xticklabels=class_labels_list,
            yticklabels=class_labels_list)

        
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix for the CNN model (For Test Set)")
plt.savefig('cm_test_cnn.png', format='png', dpi=600)


conf_matrix_2 = confusion_matrix(y_valid, valid_result)

plt.figure(figsize=(30,30))
sns.set(font_scale=5.0)
sns.heatmap(conf_matrix_2, annot=True, fmt="d", cmap="Blues", linewidths=2.8,
            annot_kws={"size": 75}, cbar_kws={'shrink': 1.0}, xticklabels=class_labels_list,
            yticklabels=class_labels_list)

        
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix for the CNN model (For Validation Set)")
plt.savefig('cm_valid_cnn.png', format='png', dpi=600)

test_f1 = f1_score(y_test, test_result, average=None)
valid_f1 = f1_score(y_valid, valid_result, average=None)



for i, label in enumerate(class_labels_list):
    print(f"F1 Score for test {label}: {test_f1[i]}")
    print(f"F1 Score for valid {label}: {valid_f1[i]}")




#[0.85284281 0.81410256 0.95165394 0.97068404]
