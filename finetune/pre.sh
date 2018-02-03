#!/bin/bash
/home/zaikun/install/caffe/build/tools/convert_imageset /home/zaikun/dl/Fine_Grained_Classification/data/cars_train/ train2.txt ../data/db/train -resize_width=227 -resize_height=227 -check_size -shuffle true
/home/zaikun/install/caffe/build/tools/convert_imageset /home/zaikun/dl/Fine_Grained_Classification/data/cars_test/ test2.txt ../data/db/test -resize_width=227 -resize_height=227 -check_size -shuffle true


/home/zaikun/install/caffe/build/tools/convert_imageset /home/zaikun/dl/Fine_Grained_Classification/data/crop_train/ train2.txt ../data/db2/train -resize_width=227 -resize_height=227 -check_size -shuffle true
/home/zaikun/install/caffe/build/tools/convert_imageset /home/zaikun/dl/Fine_Grained_Classification/data/crop_test/ test2.txt ../data/db2/test -resize_width=227 -resize_height=227 -check_size -shuffle true
