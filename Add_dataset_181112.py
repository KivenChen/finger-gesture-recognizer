''' 180922 added turned images to greyscale'''
import numpy as np
import PIL.Image as Image
import h5py
import tensorflow as tf
from time import sleep
from resnets_utils import *
from keras.preprocessing.image import array_to_img, img_to_array
# confirmed: png import successful.
# TODO 1. read from text file the opening dir of images
# TODO 2 output to x,y sets
# TODO 3: read from h5 and create new array
# TODO 4: concat the two array and update dataset
from wheels import *


blue('Loading original dataset')
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_h5_dataset('datasets nongrey')

for i in range(train_set_x_orig.shape[0]):
    imgdat = train_set_x_orig[i]
    img = array_to_img(imgdat)
    imgdat = rgb_to_grey_rgbdata(img)
    train_set_x_orig[i] = imgdat

for i in range(test_set_x_orig.shape[0]):
    imgdat = test_set_x_orig[i]
    img = array_to_img(imgdat)
    imgdat = rgb_to_grey_rgbdata(img)
    test_set_x_orig[i] = imgdat


green('Loading completed')
train_pos = open("images/train.txt")
i = 1
for s in train_pos.readlines():
    if i % 100 == 0:
        print("round "+str(i))
    (dir, num) = s.split(" ")
    num = int(num)
    image = Image.open(dir)
    imgdat = rgb_to_grey_rgbdata(image)
    imgdat = imgdat.reshape((1, 64, 64, 3))
    
    image = tf.image.random_brightness(image, 0.7)
    image = tf.image.flip_left_right(image)
    image = tf.image.random_contrast(image, 0.3, 1.0)
    imgdat2 = rgb_to_grey_rgbdata(image)
    imgdat2 = imgdat2.reshape((1, 64, 64, 3))
    
    train_set_x_orig = np.concatenate((train_set_x_orig, imgdat, imgdat2), axis=0)  # todo see if it works
    train_set_y_orig = np.concatenate((train_set_y_orig, np.array([[num]]), np.array([[num]])), axis=1)
    i += 1
print(train_set_x_orig.shape)
print(train_set_y_orig.shape)

test_pos = open("images/test.txt")
i = 1
for s in test_pos.readlines():
    if i % 100 == 0:
        print("round "+str(i))
    (dir, num) = s.split(" ")
    num = int(num)
    image = Image.open(dir)
    imgdat = rgb_to_grey_rgbdata(image)
    imgdat = imgdat.reshape((1, 64, 64, 3))
    test_set_x_orig = np.concatenate((test_set_x_orig, imgdat), axis=0)
    test_set_y_orig = np.concatenate((test_set_y_orig, np.array([[num]])), axis=1)
    i += 1

print(test_set_x_orig.shape)
print()
print(test_set_y_orig.shape)

np.save("datasets/train_x", train_set_x_orig)
np.save("datasets/train_y", train_set_y_orig)
np.save("datasets/test_x", test_set_x_orig)
np.save("datasets/test_y", test_set_y_orig)
np.save("datasets/classes", classes)

'''
test_target = h5py.File("datasets/test.h5", 'w')
train_target = h5py.File("datasets/train.h5", 'w')
print(test_target.create_dataset("test_set_x", data=np.squeeze(test_set_x_orig)))
print(test_target.create_dataset("test_set_y", data=np.squeeze(test_set_y_orig)))

train_target.create_dataset("train_set_x", data=train_set_x_orig)
train_target.create_dataset("train_set_y", data=train_set_y_orig)

test_target.create_dataset("list_classes", data=classes)

test_target.close()
train_target.close()
'''
