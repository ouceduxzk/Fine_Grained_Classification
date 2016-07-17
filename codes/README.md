Step to run the codes :

	1.install caffe, download bvlc_googlenet.caffemodel and download images into the cars_train and cars_test folder under the prepare folder 
	2. in the prepare folder, change the directory of train_path and test_path to your own path and run : python preprocessing.py, which will generate the  crop_train and crop_test
	3. sh pre.sh , which will call caffe to save the images into a database
	4. change the path in the models/finetuen_car/train_val.prototxt into the path of your db
	5. sh run.sh, the models will be saved in the models directory
	6. hyper parameters are in solver.prototxt file and you can play with that .
	
	
	
	