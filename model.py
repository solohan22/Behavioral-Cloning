"""
Steering angle prediction model
"""
import csv
import pickle
import numpy as np
import json
import cv2
import random
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, ELU, MaxPooling2D, Convolution2D
from sklearn.cross_validation import train_test_split

# load the data
data_c = csv.reader(open('driving_log.csv'), delimiter=",",quotechar='|')

img_center = []
steering = []

# store center camera images into array
for row in data_c: 
	img_center.append(row[0])
	steering.append(row[3])
img_center = np.asarray(img_center)

# directly get training steering values
y_train = np.asarray(steering, dtype=np.float32)

# channel, row and column sizes
ch, row, col = 1, 18, 80

# crop and save image in the training array
X_train=[]
for image in img_center:
	img=mpimg.imread(image)[65:135:4, 0:-1:4, 0:1]
	X_train.append(img)
X_train=np.asarray(X_train)

# split the data into 90% train and 10% test sets,
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10)

# show train and test set size
train_size = X_train.shape[0]
test_size = X_test.shape[0]

print("train size:", train_size)
print("test size:", test_size)

# shuffle the data
X_train, y_train = shuffle(X_train, y_train) 

# actual network structure
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(row, col, ch)))
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(4, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(2, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

# compile and train the model, use 15% as validation set 
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, nb_epoch=4, verbose=1, validation_split=0.15)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# save model and weights
model.save_weights('model.h5')
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())
