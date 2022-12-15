'''
-------------------------------------------------------------------------
File:       handwriting_digits_recognition.py
Project:    Digit Recognition Python
Author:     Janhavi Sathe

Description:
    Take the standard MNIST handwritten digit dataset, and as usual, split 
    it into training and testing data. Treat each image as a vector. For 
    all the test images, calculate the nearest neighbor from the training 
    data, and report this label as the prediction. 
-------------------------------------------------------------------------
Revision History:
2020-May-27	[JS]: Created
-------------------------------------------------------------------------
'''

# Import dataset-- it is available in TensorFlow's examples
from math import sqrt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(
    'D:\CCOEW\Future\Workshops\MINTs\Problems\Digit Recognition Python\MINST_data')

# Setting up variables for datasets
# :5000 Pro- full set takes a lot of time; Con- lesser accuracy
train_images = np.asarray(mnist.train.images[:5000])
train_labels = np.asarray(mnist.train.labels[:5000])
test_images = np.asarray(mnist.test.images)
test_labels = np.asarray(mnist.test.labels)


# STEP 1: Function to calculate Euclidean distance
# Can use sum() from numpy as well
def euclidean_distance(vector1, vector2):
    squared_distance = 0.0  # initialize
    # last value from each vector is ignored (related to data format) --> range = len(vector)-1
    for i in range(len(vector1) - 1):
        squared_distance += (vector1[i] - vector2[i]) ** 2
    return sqrt(squared_distance)


# STEP 2: Get nearest Neighbour --> num_neighbours = 1? hard-code 1?
#                                   >> If we always only want 1, no need to sort. while calculating
#                                      distances, keep updating a var for shortest
def nearest_neighbours(test_row, num_neighbours):
    distances = list()
    for i in range(len(train_images)):
        train_row = train_images[i]
        train_label = train_labels[i]
        distances.append(
            (train_label, euclidean_distance(test_row, train_row)))
    # Sort by second value of tuple in distance
    distances.sort(key=lambda tup: tup[1])

    neighbours = list()
    for i in range(num_neighbours):
        # Keep only labels, from position [0]
        neighbours.append(distances[i][0])
    return neighbours


# STEP 3: Make a classification prediction by counting majority in labels
def digit_prediction(train, test_row, num_neighbours):
    all_neighbours = nearest_neighbours(test_row, num_neighbours)
    closest_neighbours = [row for row in all_neighbours]
    # returns most frequent label
    prediction = max(set(closest_neighbours), key=closest_neighbours.count)
    return prediction


# Main Implementation
k = 1
correct = 0
for i in range(len(test_images)):
    test_image = test_images[i]
    output = str(digit_prediction(train_images, test_image, k))
    expected = str(test_labels[i])
    if (output == expected):
        correct += 1
    print("\nExpected: " + expected + "\tOutput: " + output)

accuracy = 100 * (correct / len(test_images))
print("Accuracy = "+accuracy+"%")

'''
Analysis:
- How accurate is this method?
    > 90% on an average
- What metric did you use for distance?
    Euclidean distance: most std metric, good distance measure to use if the input variables are similar in type
- How fast/slow is your implementation? can you analyze the bottlenecks and speed things up?
    SLOW
- Any ideas on improving accuracy?
    k-d trees?
    K-NN with non-linear deformation (IDM)      shiftable edges     0.54	Keysers et al. IEEE PAMI 2007
    K-NN with non-linear deformation (P2DHMDM)	shiftable edges     0.52	Keysers et al. IEEE PAMI 2007
-------------------------------------------------------------------------
References/Resources:
1. https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/tutorials/mnist/download/index.md
2. https://towardsdatascience.com/mnist-with-k-nearest-neighbors-8f6e7003fab7
3. https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
4. https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
5. http://yann.lecun.com/exdb/mnist/ 
-------------------------------------------------------------------------
'''
