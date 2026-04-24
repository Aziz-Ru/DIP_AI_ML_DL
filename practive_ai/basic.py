from keras.layers import Dense
from keras.models import Sequential


model = Sequential([
        Dense(4,activation = 'relu',input_shape=(8,)),
        Dense(8,activation='relu'),
        Dense(16,activation='relu'),
        Dense(10,activation='softmax')
        ])

model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary(show_trainable=True)

