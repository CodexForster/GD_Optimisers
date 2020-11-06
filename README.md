# Introducing Higher Moment Terms in Heavy-Ball Optimisation
In this project, we look forward to introduce higher "moment" terms in Heavy-Ball optimisation Studies on Optimisation Algorithms in Machine Learning and look at how they affect the training process in terms of rate of convergence, probability of convergence, etc. We call our model the HEavier-Ball Optimisation (HerBo).

We look forward to make this project an exhaustive study by focussing on one or more benchmark problems (like MNIST, to compare with existing optimisation methods), include ablative tests on hyper-parameters (like learning rate, a, b) and parameters, and thus compare how our proposed change fares with existing methods.

The "SGD.ipynb" file has the results for the CNN architecture similar to that used in the Adam paper but without Adam optimisation, i.e only plain SGD. We got a 71.65% accuracy for SGD.

NOTE: The code in this repo is an improved version of what is present in https://github.com/CNN-NISER.
Until the final version comes out, if you feel the code in this repo is not understandable, see the other link for better understanding of what is going on, and then one can come back to see what we are trying to do here. 

## Upcoming Files
 - We will include the Adam optimiser code for CNN and test it for the MNIST data.

 - We will include a feature that can run test data to get test accuracy of the model once every 'n' epochs.

 - After this we would code the proposed HerBo optimisation method in a CNN to tackle the MNIST database.
