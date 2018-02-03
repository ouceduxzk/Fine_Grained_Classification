import numpy as np
import os, cv2
from collections import defaultdict
import math

train_path = '/users/zaikun/dl/Fine_Grained_Classification/data/cars_train/'
train_files = os.listdir(train_path)
train_files = [ f for f in train_files if f.endswith('jpg')]
test_path = '/users/zaikun/dl/Fine_Grained_Classification/data/cars_test/'
test_files = os.listdir(test_path)
test_files = [ f for f in test_files if f.endswith('jpg')]
imageSize = 256

def getBox(fn):
	lines = open(fn, 'r').readlines()
	label = []
	img2box = defaultdict(list)  
	for line in lines : 
		tmp = line.strip().split()
		box = tmp[:4]
		box = [ int(math.floor(float(x))) for x in box]
		img2box[tmp[5]] = box
		#label.append(int(tmp[4]))
	#label = np.array(label)
	#np.save('y_train.npy', label)
	return img2box 

def cvresize(image, basewidth, img2box):
    img = cv2.imread(image)
    box = img2box[os.path.basename(image)]
    img = img[box[1]:box[3], box[0]:box[2],:]
    img = cv2.resize(img, (imageSize,imageSize)) 
    newpath = image.split('/')
    newpath[-2] = 'crop_' + newpath[-2][5:]
    newpath = '/'.join(newpath)
    print(newpath)
    cv2.imwrite(newpath ,img)

def image_resize(image, basewidth, img2box):
    from PIL import Image
    img = Image.open(image)
    box = img2box[os.path.basename(image)]
    box[0] = box[0] - 16
    box[1] = box[1] - 16
    box[2] = box[2] + 16
    box[3] = box[3] + 16
    img = img.crop(tuple(box))
    w, h = img.size
    r = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    r = np.array(r)
    return r 

def preprocess():
    train_box = getBox('celldata.dat')
    test_box = getBox('test_anno.txt')
    #train = np.zeros([len(train_files), 3,  imageSize, imageSize])
    for i,f in enumerate(train_files):
        cvresize(train_path + train_files[i], imageSize, train_box)

    for i,f in enumerate(test_files):
        cvresize(test_path + test_files[i], imageSize, test_box)

preprocess()
