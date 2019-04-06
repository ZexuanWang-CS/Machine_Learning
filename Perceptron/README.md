Copyright: 2019, Zexuan Wang

It implements a linear classifier using a perceptron to classify whether the data points in the UCI adult dataset have income greater or less than 50k. If it is greater than 50k, then the class is labeled as 1, and if less than 50k, then the class is labeled as -1.

To execute the file, you can choose to specify the parameters by typing the arguments. For example, '--nodev' means that the training process does not use any dev/validation files to avoid the overfitting, '--iterations [int]’ specifies the number of iteration through the full training process to perform, '--lr [float]’ specifies the learning rate to update the weights in the training process. If you do not specify the parameters, the default values will be used which are using dev/validation files, 50 iterations, and 1.0 learning rate.

The algorithm of the perceptron linear classifier is that we first initialize a weight vector that each weight corresponds to each feature. If the dot product between the weight vector and any data point is not matching its label, the weight vector will be updated. In total, there are two loops to go through: the first loop is the iteration loop, the second loop is the training dataset. The dev/validation dataset is used in the way that after 10 loops, the last five accuracies will be compared with the five value before them. If the last five accuracies start to drop, we assume we already hit the maximum and thus break the loop.

The performance of this algorithm varies with the iterations. The following two lists shows the accuracy of the test files versus iteration.
Iteration  = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]
Accuracy = [0.786, 0.799, 0.812, 0.797, 0.797, 0.781, 0.799, 0.816, 0.721, 0.786, 0.814, 0.778, 0.801]
The accuracy results are showing slight increment along the iterations, but not stable due to the simplicity of this linear classifier. The increment is because the model is trained to become better and better. The subsequent decrement is probably because the model is starting overfitting due to too many iterations, and could not generalize well in the test file.
