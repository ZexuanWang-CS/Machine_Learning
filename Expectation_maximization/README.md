Copyright: 2019, Zexuan Wang

It implements the expectation-maximization (EM) algorithm using the mixed gaussian model.

To execute the file, you can choose to specify the parameters by typing the arguments. For example, '--nodev' means that the training process does not use any dev/validation files to avoid the overfitting, '--iterations [int]’ specifies the number of iteration through the full training process to perform, '--cluster_num' specifies the number of clusters of Gaussian distribution.

Given the same cluster number, say 3, we notice the difference of model performance along different iterations.
iteration 1: loglikelyhood -4.629561
iteration 2: loglikelyhood -4.568416
iteration 3: loglikelyhood -4.543453
iteration 4: loglikelyhood -4.533566
iteration 5: loglikelyhood -4.528869
iteration 6: loglikelyhood -4.526327
iteration 7: loglikelyhood -4.524857
iteration 8: loglikelyhood -4.523901
iteration 9: loglikelyhood -4.523174
iteration 10: loglikelyhood -4.522540
iteration 11: loglikelyhood -4.521932
iteration 12: loglikelyhood -4.521305
iteration 13: loglikelyhood -4.520625
iteration 14: loglikelyhood -4.519851
iteration 15: loglikelyhood -4.518929
iteration 16: loglikelyhood -4.517785
iteration 17: loglikelyhood -4.516321
iteration 18: loglikelyhood -4.514419
iteration 19: loglikelyhood -4.512000
iteration 20: loglikelyhood -4.509182
We can see the log likelihood is increasing, indicating a better fit of the model.

Given the same iterations number, say 5, we notice the difference of model performance along different clusters.
clusters: 2 Train LL: -4.562159321567106
clusters: 5 Train LL: -4.5304355277933155
clusters: 7 Train LL: -4.488974817650457
clusters: 10 Train LL: -4.45091001335877
We can see the log likelihood is increasing, indicating a better fit of the model.
