import numpy as np
import os, cv2
from collections import defaultdict
import math
train_path = '/users/zaikun/dl/Fine_Grained_Classification/cars_train/'
train_files = os.listdir(train_path)
train_files = [ f for f in train_files if f.endswith('jpg')]
test_path = '/users/zaikun/dl/Fine_Grained_Classification/cars_test/'
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
		label.append(int(tmp[4]))
	label = np.array(label)
	#np.save('y_train.npy', label)
	return img2box 

def cvresize(image, basewidth, img2box):
        img = cv2.imread(image)
        box = img2box[os.path.basename(image)]
        img = img[box[1]:box[3], box[0]:box[2],:]
	img = cv2.resize(img, (imageSize,imageSize)) 
	#r = np.array(img).transpose(2,0,1)
        #return r 
        newpath = image.split('/')
        newpath[-2] = 'crop_' + newpath[-2][5:]
        newpath = '/'.join(newpath)
        #print(newpath)
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
        
	r = img.resize((basewidth, basewidth), Image.ANTIALIAS)
	r = np.array(r)
	return r 

def preproces():
	train_box = getBox('celldata.dat')
        testbox = getBox('test_box.txt')
	#train = np.zeros([len(train_files), 3,  imageSize, imageSize])
	#for i,f in enumerate(train_files): 
    	#	if i % 100==0 : print i
	#	cvresize(train_path + train_files[i], imageSize, train_box)
        for i,f in enumerate(test_files):
                if i % 100==0 : print i
                cvresize(test_path + test_files[i], imageSize, testbox)

	#np.save('car_train.npy', train)
preproces()
#image_resize(train_path + '00001.jpg', 224)
