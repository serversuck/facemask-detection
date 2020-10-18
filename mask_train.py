from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import os
import imutils

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score ,accuracy_score, confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

width = 128 # มิติของภาพ 128x128
height = 128
mypath = 'train/'
imagePaths = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

labels = []
rawImages = []
for imagePath in tqdm(imagePaths):

    image = cv2.imread(imagePath)
    label = imagePath.split('/')[1].split('.')[0]
    image = cv2.resize(image, (width,height))
    rawImages.append(image)
    labels.append(label)

x = np.array(rawImages)
labels = np.array(labels)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels)
y = le.transform(labels) #y คือ encoded label
print(labels[40],y[40])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

base_model = keras.applications.MobileNetV2(input_shape=(width,width,3),
                                                   include_top=False,
                                                   weights='imagenet')
base_model.trainable = False
num_classes = 2
model = keras.Sequential([
        base_model,
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

# model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
#               loss='sparse_categorical_crossentropy',
#               metrics= ['accuracy'])
bz = 32
ep = 5
#

base_model.trainable=True
for layer in base_model.layers[:100]:
  layer.trainable = False
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001), loss='sparse_categorical_crossentropy'
                , metrics= ['accuracy'])

history = model.fit(X_train ,y_train ,batch_size=bz, epochs=ep ,validation_data=(X_test,y_test) )
model.save("mobileV2-maskcheckx.h5")

