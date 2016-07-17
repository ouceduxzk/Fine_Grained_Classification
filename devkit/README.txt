This file gives documentation for the cars 196 dataset.
(http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

----------------------------------------
Metadata/Annotations
----------------------------------------
Descriptions of the files are as follows:

-cars_meta.mat:
  Contains a cell array of class names, one for each class.

-cars_train_annos.mat:
  Contains the variable 'annotations', which is a struct array of length
  num_images and where each element has the fields:
    bbox_x1: Min x-value of the bounding box, in pixels
    bbox_x2: Max x-value of the bounding box, in pixels
    bbox_y1: Min y-value of the bounding box, in pixels
    bbox_y2: Max y-value of the bounding box, in pixels
    class: Integral id of the class the image belongs to.
    fname: Filename of the image within the folder of images.

-cars_test_annos.mat:
  Same format as 'cars_train_annos.mat', except the class is not provided.

----------------------------------------
Submission file format
----------------------------------------
Files for submission should be .txt files with the class prediction for
image M on line M. Note that image M corresponds to the Mth annotation in
the provided annotation file. An example of a file in this format is
train_perfect_preds.txt

Included in the devkit are a script for evaluating training accuracy,
eval_train.m. Usage is:

(in MATLAB)
>> [accuracy, confusion_matrix] = eval_train('train_perfect_preds.txt')

If your training predictions work with this function then your testing
predictions should be good to go for the evaluation server, assuming
that they're in the same format as your training predictions.
