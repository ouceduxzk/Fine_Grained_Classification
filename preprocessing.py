import numpy as np
import os
from PIL import Image
from collections import defaultdict
import math
#from skimage.filters import sobel 
train_path = './cars_train/'
train_files = os.listdir(train_path)
train_files = [ f for f in train_files if f.endswith('jpg')]

def getBox():
	lines = open('celldata.dat', 'r').readlines()
	label = []
	img2box = defaultdict(list)  
	for line in lines : 
		tmp = line.strip().split()
		box = tmp[:4]
		box = [ int(math.floor(float(x))) for x in box]
		img2box[tmp[5]] = box
		label.append(int(tmp[4]))
	label = np.array(label)
	np.save('y_train.npy', label)
	return img2box 


def image_resize(image, basewidth, img2box):
        img = Image.open(image).convert('L')
        box = img2box[os.path.basename(image)]
        box[0] = box[0] - 16
        box[1] = box[1] - 16
        box[2] = box[2] + 16
        box[3] = box[3] + 16
        img = img.crop(tuple(box))
        w, h = img.size
	#img = np.asarray(img.getdata(),dtype=np.float32).reshape((img.size[0],img.size[1])
	# if w > h : 
	# 	npad = ( (0,0), (w- h, 0))
	# 	img = np.pad(img, npad, 'constant' )
	# else:
	# 	npad = ((h-w, 0), (0,0))
	# 	img = np.pad(img, npad, 'constant')
	# newImage = Image.fromarray(img)
	#r = newImage.resize((basewidth, basewidth), Image.ANTIALIAS)
	#r = np.array(r)/255.0
	r = img.resize((basewidth, basewidth))
	r = np.array(r)
	return r 

def preproces():
	img2box = getBox()
	train = np.zeros([len(train_files), 224,224])
	for i,f in enumerate(train_files): 
    		if i % 100==0 : print i
		train[i] = image_resize(train_path + train_files[i], 224, img2box)
	print train.shape
	np.save('car_train.npy', train)

#image_resize(train_path + '00001.jpg', 224)
