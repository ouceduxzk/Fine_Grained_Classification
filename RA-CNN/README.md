Reproduce the experiments of RA-CNN on stanford cars dataset


Steps to reproduce the paper：

1. define a super networks with three subnets inside, initialize three subnets (n1, n2, n3) of RA-CNN with the same pretrained VGG-19 model.

2. Use the last conv layer of the n1 (here conv5_4) to find the max repsonse p (x0, y0) and define a bbox center of p with half legnth of original images, this is kind of a label for training APN (apn1). Similarly, use the croped image from apn1 as input for n2 and get the bbox for apn2 and train apn2 (while fixing the classificaiton network)

3. Once the apn1, apn2 is trained, fix the weights in apn to training the conv/classification network util converging.

4. Alternative training of classification network and APN network (fixing the others lr to be 0) untile both of the nets converge.


##Features 
- [x] load the same weights into  three subnets with three differenet names of blobs (through python inferface, save the save weights into three sets of names)
- [] Train the APN network with pool5 (512 max response numbers) to supervise learning (tx, ty, tl) 
- [] Alternative training