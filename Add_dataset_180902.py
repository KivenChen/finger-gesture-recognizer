''' 180922 added turned images to greyscale'''
import numpy as np
import PIL.Image as Image
import h5py
from time import sleep
from resnets_utils import *
# confirmed: png import successful.
# TODO 1. read from text file the opening dir of images
# TODO 2 output to x,y sets
# TODO 3: read from h5 and create new array
# TODO 4: concat the two array and update dataset
from wheels import *
blue('Loading original dataset')
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_h5_dataset('datasets nongrey')

print(train_set_x_orig.shape)
green('Loading completed')
train_pos = open("./images/train.txt")
i = 1
result=[train_set_x_orig,]
for s in [part for part in train_pos.read().split('./')if part != '']:
    if i % 100 == 0:
        print("round "+str(i))
    dir, num = s.split(" ")
    num = int(num)
    try:
        image = Image.open(dir)
    except:
        continue
    imgdat = np.asarray(image)
    imgdat = imgdat.reshape((1, 64, 64, 3))
    result.append(imgdat)
    train_set_y_orig = np.concatenate((train_set_y_orig, np.array([[num]])), axis=1)
    i += 1
import time
time.sleep(1)
train_set_x_orig = np.concatenate(result, axis=0)  # todo see if it works
print(train_set_x_orig.shape)
print(train_set_y_orig.shape)

result = [test_set_x_orig,]
test_pos = open("images/test.txt")
i = 1
for s in [part for part in test_pos.read().split('./')if part != '']:
    if i % 100 == 0:
        print("round "+str(i))
    dir, num = s.split(" ")
    num = int(num)
    try:
        image = Image.open(dir)
    except:
        continue
    imgdat = np.asarray(image)
    imgdat.resize((1,64,64,3))
    result.append(imgdat)
    test_set_y_orig = np.concatenate((test_set_y_orig, np.array([[num]])), axis=1)
    i += 1
test_set_x_orig = np.concatenate(result, axis=0)
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