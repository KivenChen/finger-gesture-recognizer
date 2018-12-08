from wheels import *
blue("Loading ResNet components ... ")

import numpy as np
import random
from wheels import *
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from resnets_utils import *
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import keras.backend as K
green("Completed")



blue("Loading model ...")
model = load_model("models\\181207\\accu0.9866666658719381.h5")
green("Completed")


def findnumber(result):
    return K.get_session().run(K.argmax(result))


def judgeone():
    path = 'C:\\Users\\java1\\Dropbox\\ResNet test\\'
    img_list = [path + i for i in justfilenames(path) if ".jpg" in i]
    for img_path in img_list:
        img = image.load_img(img_path, target_size=(64, 64)).rotate(-90)
        # img = img.convert('L')
        # one_channel = np.asarray(img)
        # x = np.concatenate((one_channel, one_channel, one_channel), axis=0)
        # x = x.reshape((1, 64, 64, 3))

        x = np.asarray(img)
        # x = preprocess_input(x)
        imshow(image.array_to_img(x))
        x = x.reshape((1, 64, 64, 3)) / 255
        print("ahahaha")
        # x = np.divide(x, 255)
        print('Input image shape:', x.shape)
        #  imshow(my_image)
        print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
        print([str("%.3f" % i) for i in np.squeeze(model.predict(x))])
        plt.show()


def check_repetitions(type_idx=0, origin_idx=0, from_input=True):
    if from_input:
        type_idx = int(input("Input: 0-Xtrain, 1-Ytrain, 2-Xtest, 3-Ytest, 4-classes\n"))
        origin_idx = int(input("Input i so the i-th item will be seen as the origin of comparison\n"))
    data = load_h5_dataset("datasets nongrey")
    origin = data[type_idx][origin_idx]
    counter = 0
    for i in data[type_idx]:
        if all(i == origin):
            counter += 1
    print("found ", counter-1, "repetitions")


def judge_test_set(specific_class=None, result_match_str=None):
    _, _, x, y, _ = load_npy_dataset("datasets nongrey")
    y = y.reshape(-1)
    for n, i in enumerate(x):
        if specific_class and specific_class != y[n]:
            continue
        result = model.predict(i.reshape((1, 64, 64, 3)) / 255)
        to_print = [str("%.3f" % p) for p in np.squeeze(result)]
        if result_match_str and result_match_str not in str(to_print):
            continue
        print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
        print(to_print)
        plt.figure(clear=True)
        plt.imshow(image.array_to_img(i))
        plt.show()


n = 100


def show_random_test_example():
    plt.figure()

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_npy_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # print(model.evaluate(X_test, Y_test))

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
    plt.figure()

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_npy_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # print(model.evaluate(X_test, Y_test))

    test = [0,0,0,0,0,0]
    train = test.copy()
    for i in Y_train:
        train[np.argmax(i)] += 1
    for i in Y_test:
        test[np.argmax(i)]+=1
    return test, train


if __name__== "__main__":
    # judge_test_set(specific_class=None, result_match_str='0.4') / exit()
    # check_repetitions() / exit()
    while True:
        main()
        # judge_test_set(4)
        # main()
    count()
