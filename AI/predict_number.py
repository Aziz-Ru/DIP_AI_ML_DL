import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# Load and preprocess the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels (0 to 9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),         # Input layer
    Dense(128, activation='relu'),         # Hidden layer 1
    Dense(64, activation='relu'),          # Hidden layer 2
    Dense(10, activation='softmax')        # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)



# Pick one test image

img = np.ones((28,28),dtype=np.uint8)


# Predict
prediction = model.predict(img.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
print("Predicted digit:", predicted_label)

plt.imshow(img, cmap='gray')
plt.show()