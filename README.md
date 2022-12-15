# mnist-digits-recognition-knn (May, 2020)

Using the MNIST Dataset, script for digit recognition using the k-nearest neighbours algorithm

## Problem Statement
Take the standard MNIST handwritten digit dataset, and as usual, split 
it into training and testing data. Treat each image as a vector. For 
all the test images, calculate the nearest neighbor from the training 
data, and report this label as the prediction.

## Steps
1. Import dataset-- it is available in TensorFlow's examples
2. Divide data into training & test data
3. Write a function to calculate Euclidean distance
4. Get nearest Neighbour
5. Make a classification prediction by counting majority in labels

## Analysis:
- How accurate is this method?
    > 90% on an average
- What metric did you use for distance?
    > Euclidean distance: most std metric, good distance measure to use if the input variables are similar in type
- How fast/slow is your implementation? can you analyze the bottlenecks and speed things up?
    > SLOW
- Any ideas on improving accuracy?
   > k-d trees?  
   > K-NN with non-linear deformation (IDM)     | shiftable edges  |   0.54	Keysers et al. IEEE PAMI 2007  
   > K-NN with non-linear deformation (P2DHMDM)	| shiftable edges  |   0.52	Keysers et al. IEEE PAMI 2007  
-------------------------------------------------------------------------
## References/Resources:
1. https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/tutorials/mnist/download/index.md
2. https://towardsdatascience.com/mnist-with-k-nearest-neighbors-8f6e7003fab7
3. https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
4. https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
5. http://yann.lecun.com/exdb/mnist/ 
-------------------------------------------------------------------------

>*Note*: Uploading in 2022 as a backup
