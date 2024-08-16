import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential

# Load the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255

# Print the shape of the training data
# print(f"Training data shape: {X_train.shape}")

# # Visualize a few examples from the dataset
# num_images = 5  # Number of images to display
# plt.figure(figsize=(10, 5))

# for i in range(num_images):
#     plt.subplot(1, num_images, i + 1)
#     plt.imshow(X_train[i], cmap='gray')
#     plt.title(f"Label: {y_train[i]}")
#     plt.axis('off')

# plt.show()

# Reshape the data
X_train_reshape = X_train.reshape(len(X_train), 28*28)
X_test_reshape = X_test.reshape(len(X_test), 28*28)
# print(X_test_reshape.shape)
# print(X_train[0])
# plt.matshow(X_train[0])
# plt.show()

# Define the model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Train the model
model.fit(X_train_reshape, y_train, epochs=5)  # Train on X_train_reshape and y_train

# Evaluate the model
model.evaluate(X_test_reshape, y_test)

# Make predictions
y_predicted = model.predict(X_test_reshape)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Visualize a test image
plt.matshow(X_test[1])
plt.show()

# Print the prediction for the test image
print(y_predicted[1])

# Compute and visualize the confusion matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

cm = cm.numpy()

plt.figure(figsize=(10, 8))
plt.matshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
