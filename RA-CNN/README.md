Reproduce the experiments of RA-CNN on stanford cars dataset


Steps to reproduce the paper：

1. define three networks, initialize three networks (n1, n2, n3) of RA-CNN with the same pretrained VGG-19 model.

2. Use the last conv layer of the n1 (here conv5_4) to find the max repsonse p (x0, y0) and define a bbox center of p with half legnth of original images, this is kind of a label for training APN (apn1). Similarly, use the croped image from apn1 as input for n2 and get the bbox for apn2 and train apn2 (while fixing the classificaiton network)

3. Once the apn1, apn2 is trained, fix the weights in apn to training the conv/classification network util converging.

4. Alternative training of classification network and APN network (fixing the others lr to be 0) untile both of the nets converge.



Note :

1. The step2 is most important since it serves the basis for the whole alternative training to work. Basically, given each input each image i, extract p' (coordiate on original image) based on the p(x0, y0) on conv5_4 and save the bbox label （tx, ty, tl） for training apn1, do the similar thing for apn2. 
