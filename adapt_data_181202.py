from kreprocessing import *
from resnets_utils import *
import logging as log
from time import time
from wheels import *

fmt = "%(asctime)-15s  %(levelname)s, %(filename)s, %(lineno)d ,  %(process)d : %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = log.Formatter(fmt, datefmt)
log.basicConfig(filename='finger.log', level=log.INFO, format=fmt)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_npy_dataset("datasets nongrey")
trainsets = []

trainset = [X_train_orig / 255]
log.info("Preprocessing begins")
for i in range(5):
    time_st = time()
    green(str(i)+" th preprocessing")
    trainset.append(random_img_preproc_pil(X_train_orig) / 255)
    log.info(str(i)+" th preprocessing complete. Duration (seconds): "+str( (time()-time_st) // 1))
# Standardize data to have feature values between 0 and 1.
np.save("datasets\\trainset", trainset)