import tensorflow as tf
from keras.api.datasets import fashion_mnist
from keras.api import *

data = fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# normalizing the image
# ensures every pixel is a value between 0 & 1
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = Sequential(
    [
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPool2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(training_images, training_labels, epochs=50)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
