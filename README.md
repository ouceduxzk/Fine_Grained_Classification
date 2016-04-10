# Fine_Grained_Classification

Task : Classification of 196 classes of cars with less than 9k images for training. It's intend for replication of the work  : 

	Monza: Image Classification of Vehicle Make and Model Using Convolutional Neural Networks and Transfer Learning

1. use pre-trained googlenet
2. train with normal rates
3. Achieved 87% top1 accuracy and 97% top5 acc with 10000 iterations


Lessons learned : 

	1.  Training from scratch is very hard with less data and more class.
	2.  Bounding box of the car really helps
	3.  fine tuning need to select a good lr and make the bottom layers learn slow, but the last few layers learn fast.