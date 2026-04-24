
from keras.layers import Dense,Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(512,activation='relu'),
        Dense(256,activation='relu'),
        Dense(128,activation='relu'),
        Dense(64,activation='relu'),
        Dense(10,activation='softmax'),
        ])
        

(x,y),(xts,yts) = mnist.load_data()

xt=x[:50000]
yt=y[:50000]
xv=x[50000:]
yv=y[50000:]

model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

tm = model.fit(xt,yt,epochs=1,batch_size=32,validation_data=(xv,yv),verbose=1)

loss,acc = model.evaluate(xts,yts,verbose=1)

x_input = xts[100].reshape(1,28,28)
pred = model.predict(x_input)

pred_label = np.argmax(pred)


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.imshow(xts[100])
plt.title(f"Label {pred_label}")

plt.show()

