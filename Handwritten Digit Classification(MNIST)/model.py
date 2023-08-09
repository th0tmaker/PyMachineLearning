import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Disable Tensorflow debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

######################
# PART I <PREPARE DATA>
######################

# Call the MNIST image classification data API (60.000 training & 10.000 testing images)
mnist_digits_dataset = tf.keras.datasets.mnist  # 28x28 images of handwritten digits from 0-9

# Unpack and save the data into two sample datasets (one for training, other for testing)
(x_train, y_train), (x_test, y_test) = mnist_digits_dataset.load_data()  # Loaded as a numpy.ndarray object

# Check training dataset array shape
# print x_train.shape <- Output: (60000, 28, 28)
# print y_train.shape <- Output: (10000, 28, 28)

# View first dataset image
plt.imshow(x_train[0], cmap='gray')
# plt.show() <- uncomment to display image in new window

# Cast the data into a more memory efficient float32 dtype
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data by scaling the images from value range [0, 255] to [0, 1]
x_train = tf.keras.utils.normalize(x_train, axis=1)  # Alternatively: x_train = x_train / 255 (1 channel: 0 to 255 b/w)
x_test = tf.keras.utils.normalize(x_test, axis=1)  # Alternatively: x_test = x_test / 255 (1 channel: 0 to 255 b/w)

# Reshape dataset array by adding an extra dimension (total: 4D) in order to perform Convolution operations with keras
IMG_SIZE = 28
x_train_rs = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test_rs = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Check the reshaped training dataset array
# print x_train:rs.shape <- Output: (60000, 28, 28, 1)
# print y_train:rs.shape <- Output: (10000, 28, 28, 1)

######################
# PART II <BUILD MODEL>
######################

# Build the model architecture in Sequential order (using multinomial logistic regression)
model = tf.keras.Sequential()

# First Convolution Layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=x_train_rs.shape[1:]))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Second Colvolution Layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Third Convolution Layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Fully Connected Layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation("relu"))

# Output Layer
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation("softmax"))

######################
# PART III <CHECK MODEL SUMMARY>
######################
# Summary of the model architecture

model.summary()

######################
# PART VI <COMPILE, TRAIN AND TEST MODEL>
######################

# Compiling the model (specifying the optimizer, loss, and metrics)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate model effectiveness on test data and save loss & accuracy
test_loss, test_acc = model.evaluate(x_test_rs, y_test)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_acc)

# Predict the output of a given (reshaped) test image from thedataset
prediction = model.predict(x_test_rs)
print(np.argmax(prediction[0]))

# Save and export model into a .keras file
# model.save("HWDigitsClassifier.keras")
