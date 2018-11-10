from wheels import *
blue("Loading ResNet components ... ")

import numpy as np
import random
from wheels import *
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import keras.backend as K
green("Completed")



blue("Loading model ...")
model = load_model("092218 152 th iter, accu 96.04")
green("Completed")

plt.figure()

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_new_dataset()


# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# print(model.evaluate(X_test, Y_test))

def findnumber(result):
    return K.get_session().run(K.argmax(result))

def judgeone():
    path = 'C:\\Users\\java1\\Dropbox\\ResNet test\\'
    img_list = [path + i for i in justfilenames(path) if ".jpg" in i]
    for img_path in img_list:
        img = image.load_img(img_path, target_size=(64, 64))
        img = img.convert('L')
        imshow(img)
        one_channel = np.asarray(img)
        x = np.concatenate((one_channel, one_channel, one_channel), axis=0)
        x = x.reshape((1, 64, 64, 3))
        print(x.dtype)
        # x = preprocess_input(x)
        x = np.divide(x, 255)
        print('Input image shape:', x.shape)
        my_image = scipy.misc.imread(img_path)
        #  imshow(my_image)
        print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
        print([str("%.3f" % i) for i in np.squeeze(model.predict(x))])
        print(findnumber(model.predict(x)))
        plt.show()


n = 100


def show_random_test_example():
    n = random.randint(1, 5000)
    imshow(image.array_to_img(X_train[n]))
    plt.show()
    blue("Y: " + str(Y_train[n]))
    n += 1

def main():
    global n
    blue("Do you want to reload test image?")
    if input().lower()=='y':
        judgeone()
    else:
        K.get_session().close()
        exit()

def count():
    test = [0,0,0,0,0,0]
    train = test.copy()
    for i in Y_train:
        train[np.argmax(i)]+=1
    for i in Y_test:
        test[np.argmax(i)]+=1
    return test, train

if __name__=="__main__":
    while True:
        main()
    count()