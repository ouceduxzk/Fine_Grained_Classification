#!/bin/bash
/home/zaikun/caffe/build/tools/convert_imageset /home/zaikun/dl/Fine_Grained_Classification/crop_train/ train2.txt db/train -resize_width=227 -resize_height=227 -check_size -shuffle true
/home/zaikun/caffe/build/tools/convert_imageset /home/zaikun/dl/Fine_Grained_Classification/crop_test/ test2.txt db/test -resize_width=227 -resize_height=227 -check_size -shuffle true
