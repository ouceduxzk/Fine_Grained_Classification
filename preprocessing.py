import numpy as np
import os
from PIL import Image
#from skimage.filters import sobel 
train_path = './cars_train/'
train_files = os.listdir(train_path)
train_files = [ f for f in train_files if f.endswith('jpg')]

def preprocessing(image, basewidth):
	
	img = Image.open(image).convert('L')
	img = np.asarray(img.getdata(),dtype=np.float32).reshape((img.size[0],img.size[1]))
	#img = np.asarray(img)
	#try :
        w, h = img.shape
	#except :
	#	img = np.asarray([img, img, img]).transpose(1,2,0)
	#	w, h , x= img.shape
	if w > h : 
		npad = ( (0,0), (w- h, 0))
		img = np.pad(img, npad, 'constant' )
	else:
		npad = ((h-w, 0), (0,0))
		img = np.pad(img, npad, 'constant')
	newImage = Image.fromarray(img)
	r = newImage.resize((basewidth, basewidth), Image.ANTIALIAS)
	r = np.array(r)/255.0
 	r = sobel(r.astype(np.float32))
	return r 

train = np.zeros([len(train_files), 224,224])
for i,f in enumerate(train_files): 
        if i % 100==0 : print i
	train[i] = image_resize(train_path + train_files[i], 224)

print train.shape
np.save('car_train.npy', train)
#image_resize(train_path + '00001.jpg', 224)
