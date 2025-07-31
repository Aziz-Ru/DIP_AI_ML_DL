from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, input_dim=2, activation='relu'),   # Input layer (2) → Hidden 1
    Dense(32, activation='relu'),                # Hidden 2
    Dense(16, activation='relu'),                # Hidden 3
    Dense(8, activation='relu'),                 # Hidden 4
    Dense(1, activation='sigmoid')               # Output layer
])

model.summary()
