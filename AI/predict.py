import numpy as np
import matplotlib.pyplot as plt

# Pick one test image

img = x_test[0]
plt.imshow(img, cmap='gray')
plt.show()

# Predict
prediction = model.predict(img.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
print("Predicted digit:", predicted_label)
