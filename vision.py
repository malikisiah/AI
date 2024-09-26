import tensorflow as tf
import keras


data = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# normalizing the image
# ensures every pixel is a value between 0 & 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# the neural network
# input layer -> 128 neurons -> output layer
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# categorical loss functino is optimal for picking categories
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("Training the Model")
model.fit(training_images, training_labels, epochs=5)
print("Training Complete")

model.evaluate(test_images, test_labels)
