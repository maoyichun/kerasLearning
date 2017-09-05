# -*- coding: utf-8 -*-

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
# from keras.utils import np_utils

# 生成训练数据，100张图片作为训练样本，图片尺寸100*100*3
x_train = np.random.random((100, 100, 100, 3))
# 生成训练标签，10个类别, 100*10
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# 生成测试数据，20张图片作为测试集
x_test = np.random.random((20, 100, 100, 3))
# 生成测试标签
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
model.fit(x_train, y_train, batch_size=32, epochs=100)
score = model.evaluate(x_test, y_test)
