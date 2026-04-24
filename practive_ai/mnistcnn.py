from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


(x,y),(xt,yt) = mnist.load_data()

train_x = x.reshape(-1,28,28,1).astype(np.float32)/255.0
test_x = xt.reshape(-1,28,28,1).astype(np.float32)/255.0




model = Sequential([
        Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        
        Conv2D(256,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        
        Flatten(),
        Dense(512,activation='relu'),
        Dense(256,activation='relu'),
        Dense(128,activation='relu'),
        Dense(64,activation='relu'),
        Dense(10,activation='softmax'),
        ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_x,y,epochs=1,batch_size=32,validation_split=0.01)

test_loss,test_acc = model.evaluate(test_x,yt)

rn = np.random.randint(10,500,size=10)
plt.figure(figsize=(8,6))
for i in range(10):
    ind = rn[i]
    input_img = test_x[ind]
    input_x =input_img.reshape(1,28,28,1)
    pred= model.predict(input_x)
    label = np.argmax(pred)
    plt.subplot(4,3,i+1)
    plt.imshow(input_img)
    plt.title(f"Label {label}")

plt.show()

