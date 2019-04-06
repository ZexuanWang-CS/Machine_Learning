Copyright: 2019, Zexuan Wang

It implements the backpropagation algorithm through an one-hidden-layer neuron network to classify whether the data points in the UCI adult dataset have income greater or less than 50k. If it is greater than 50k, then the class is labeled as 1, and if less than 50k, then the class is labeled as 0.

To execute the file, you can choose to specify the parameters by typing the arguments. For example, '--nodev' means that the training process does not use any dev/validation files to avoid the overfitting, '--iterations [int]’ specifies the number of iteration through the full training process to perform, '--lr [float]’ specifies the learning rate to update the weights in the training process, ‘--weights_files’ specifies the initial weights for the model if not using the randomly generated weights, ‘--hidden_dim’ specifies the number of nodes in the hidden layer, '--print_weights' will print the final weights. If you do not specify the parameters, the default values will be used which are using dev/validation files, 1 iteration, 0.01 learning rate, 5 nodes in the hidden layer, and not print out weights.

The backpropagation algorithm of the neuron network is that we first initialize some weight arrays with each corresponding to a node in the previous layer. During the feed-forward process, the dot products between the input and the weight array will be used as the input for the next layer. This iterative feed-forward process continues until the last output layer. Then the errors are calculated between the expected values and the real outputs. The gradients between the total error and each weight guides the back propagation process to update all the weights. The dev/validation dataset is used in the way that after 10 loops, the last five accuracies will be compared with the five value before them. If the last five accuracies start to drop, we assume we already hit the maximum and thus break the loop.

The performance of this algorithm varies with the iterations. The following two lists shows the accuracy of the test files versus iteration.
Iteration  = [1, 5, 10, 15, 20, 25, 50]
Accuracy = [0.7568, 0.8368, 0.8471, 0.8491, 0.8515, 0.8517, 0.8503]
The accuracy results are showing fast increment along the first several iterations, and slightly decrease due to the overfitting to the training data, and could not generalize well in the test file.

The performance of this algorithm varies with the learning rates. The following two lists shows the accuracy of the test files versus learning rate.
Learning Rate  = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
Accuracy = [0.7568, 0.8361, 0.8454, 0.8472, 0.8471, 0.8466, 0.8361]
The accuracy results are showing fast increment among small learning rates, and slightly decrease due to bypassing the global minimum due to large learning rates.
