import tflearn
from tflearn import conv_2d,max_pool_2d,fully_connected,add_weights_regularizer,\
    input_data,regression,DNN,dropout
import cv2
import time
import numpy as np
import glob

X=[]
Y=[]


print('Loading dataset...')
for cat in glob.glob('resized/cat*.jpg'):
    img = cv2.imread(cat)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = np.float16(np.abs(np.subtract(np.divide(img, 255), 1)))
    X.append(img)
    Y.append([1,0])
for dog in glob.glob('resized/dog*.jpg'):
    img = cv2.imread(dog)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.float16(np.abs(np.subtract(np.divide(img, 255), 1)))
    X.append(img)
    Y.append([0,1])
print("Done")

X = np.array(X).reshape(-1,150,150,1)
print(len(Y))

model = input_data([None,150,150,1])
model = conv_2d(model, 30, 5, activation='relu',regularizer='L2')
model = max_pool_2d(model, 5)

model = conv_2d(model, 60, 5, activation='relu')
model = max_pool_2d(model, 5)

model = conv_2d(model, 100, 5, activation='relu',regularizer='L2')
model = max_pool_2d(model, 5)

model = conv_2d(model, 60, 5, activation='relu')
model = max_pool_2d(model, 5)

model = fully_connected(model, 1000, activation='relu')
model = dropout(model, 0.9)

model = fully_connected(model, 2, activation='softmax')
model = regression(model, loss='categorical_crossentropy', name='targets')
model = DNN(model)

# model.load('ai')
#
# print("Ready")
# while True:
#     num = input()
#     image = cv2.imread('test1/{}.jpg'.format(num))
#     image = cv2.resize(image, (150, 150))
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     image = np.float16(np.abs(np.subtract(np.divide(image, 255), 1)))
#     X = np.array(image).reshape(-1, 150, 150, 1)
#     print(model.predict(X))
print("Begin training...")
start = time.time()
model.fit(X,Y,n_epoch=10,show_metric=True,shuffle=True,validation_set=0.15)
print("Total time taken:",time.time()-start,'seconds')