# -*- coding: utf-8 -*-

import time
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
# from keras.layers import Dropout, Flatten, Dense
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.models import Sequential
from keras.preprocessing import image
import numpy as np

t0 = time.time()    # 记录时间
model = VGG16(weights='imagenet')   # 224*224
img_path = 'testpics/car.jpg'
img = image.load_img(img_path, target_size=(224, 224))  # 224*224
x = image.img_to_array(img)  # 三维(3,224,224)
x = np.expand_dims(x, axis=0)   # 四维(1,3,224,224)
x = preprocess_input(x)
y_pred = model.predict(x)
print('测试图:', decode_predictions(y_pred))   # 输出5个最高概率排名
print('耗时: %.2f seconds' % (time.time() - t0))

# # Load image
# t0 = time.time()
# height, width = (224, 224)
# img_path = 'Tulips.jpg'
# img = image.load_img(img_path, target_size=(224, 224))  # 224*224
# x = image.img_to_array(img)   # 三维(3,224,224)
# x = np.expand_dims(x, axis=0)   # 四维(1,3,224,224)
# x = preprocess_input(x)
# print('训练样例:', x.shape)
# print('取数据耗时: %.2f seconds.' % (time.time() - t0))
#
# # Train VGG16
# print('开始建模...')
# model = Sequential()
#
# # Block1, 2layers
# model.add(Convolution2D(64, 3, 3, activation='relu',
#                         border_mode='same', input_shape=(3, height, width)))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# # Block2, 2layers
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# # Block3, 3layers
# model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
# model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
# model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# # Block4, 3layers
# model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
# model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
# model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# # Block5, 3layers
# model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
# model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
# model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# # Classification block, Dense 3layers
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1000, activation='softmax'))
# model.load_weights('vgg16_weights.h5')
# print("建模CNN完成 ...")
# test master hehe
