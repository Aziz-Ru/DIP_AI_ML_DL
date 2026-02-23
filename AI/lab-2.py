from keras.layers import Input,Dense,Flatten
from keras.models import Model

inputs=Input((18,18))
flatten_layer=Flatten()(inputs)
hidden_layer1=Dense(2,activation='relu',name='hidden_layer1')(flatten_layer)
hidden_layer2=Dense(4,activation='relu',name='hidden_layer2')(hidden_layer1)
hidden_layer3=Dense(8,activation='relu',name='hidden_layer3')(hidden_layer2)
hidden_layer4=Dense(16,activation='relu',name='hidden_layer4')(hidden_layer3)
hidden_layer5=Dense(8,activation='relu',name='hidden_layer5')(hidden_layer4)
hidden_layer6=Dense(4,activation='relu',name='hidden_layer6')(hidden_layer5)
hidden_layer7=Dense(2,activation='relu',name='hidden_layer7')(hidden_layer6)

outputs=Dense(1,activation='softmax',name='output')(hidden_layer7)

model=Model(inputs,outputs)

model.summary(show_trainable=True)
