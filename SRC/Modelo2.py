import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Deconvolution2D, MaxPooling2D, Flatten
from keras import layers as L

path = './Euro_Sat/'

dic = {}
entrada = []
salida = []
i = 0

for ele in os.listdir(path):
    if os.path.isdir(path+ele):
        for file in os.listdir(path+ele):
            im = Image.open(os.path.join(path,ele,file))
            im_array = np.array(im)
            entrada.append(im_array)
            salida.append(i)
        dic[i] = ele
        i += 1

clases_salida = to_categorical(salida)

entrada = np.array(entrada)
entrada = entrada/255

x_train, x_test, y_train, y_test = train_test_split(entrada, clases_salida, test_size=0.15, shuffle=True)

in_shape = x_train[0,:,:,:].shape

model = Sequential()

model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=in_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(len(dic), activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25)

