
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


model = Sequential([
        Dense(4,activation = 'relu',input_shape=(1,)),
        Dense(8,activation='relu'),
        Dense(16,activation='relu'),
        Dense(1,activation='linear')
        ])


def eq1(x):
    return 5*x**2 + 10*x -5


x = np.linspace(-10,10,2000).reshape(-1,1)
y = eq1(x)

index = np.random.permutation(len(x))
x= x[index]
y= y[index]

train_x=x[:1600]
train_y=y[:1600]

test_x=x[1600:]
test_y=y[1600:]


model.compile(optimizer = 'adam',loss='mse',metrics=['mae'])

h= model.fit(train_x,train_y,verbose=1,epochs=500,batch_size=32,validation_split=0.01)
print("train complete")
loss,mae = model.evaluate(test_x,test_y,verbose=1)

print(loss)
print(mae)
# xp = np.array([[2]])
# yp = model.predict(xp)
#print(f"For : {xp[0][0]} predict : {yp[0][0]}")
y_pred = model.predict(test_x)


plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
plt.scatter(test_x,test_y,label="Accurate",color='blue')
plt.scatter(test_x,y_pred,label='Guess',color='red')


plt.subplot(2,2,2)
plt.plot(h.history['loss'],label='Train Label')
plt.show()
# model.summary(show_trainable=True)
