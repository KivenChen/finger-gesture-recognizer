from PIL import Image
import numpy as np

from wheels import *


def just_folder_names(dir):  # search all files in the dir
	onlyfolders = [f for f in listdir(dir) if not isfile(join(dir, f))]
	return onlyfolders


def _get_classes(dir='./'):
	try:
		return [fname.split(' ') for fname in just_folder_names(dir)]
	except:
		err("folder names must be in '<num> <name>' patterns")
		exit()


def _imgs_to_ndarray(imglist, tosize=64):
	dataset = np.array([])
	for i in imglist:
		img = Image.open(i)
		img.load()
		resizedimg = img.resize((tosize, tosize), Image.ANTIALIAS)
		data = np.reshape(np.asarray(resizedimg), newshape=(1, tosize, tosize, 3))
		dataset = np.append(dataset, data, axis=0)
	return dataset


def read_from_class_folders_and_convert(src, todir='datasets/', tosize=64, test_or_train='train', save_classes=False):
	classes = _get_classes(src)
	x = np.array([])
	y = np.array([])
	if classes is None:
		exit()
	for (code, name) in classes:
		section = join(src, code+' '+name)
		targets = justfilenames(section)
		if x.shape == (0,):
			x = _imgs_to_ndarray(targets, tosize)
			y = np.ones((len(targets))) * code
		else:
			x_toconcat = _imgs_to_ndarray(targets, tosize)
			y_toconcat = np.ones((len(targets))) * code
			x = np.append(x, x_toconcat, axis=0)
			y = np.append(y, y_toconcat, axis=0)
	np.save(x, test_or_train+"_x")
	np.save(y, test_or_train+"_y")
	if save_classes:
		classes_arr = np.array([name for name in (code, name) in classes])
		np.save(classes_arr, "classes")


if __name__ == "__main__":
	pass
