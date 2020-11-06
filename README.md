# Introducing Higher Moment Terms in Heavy-Ball Optimisation
In this project, we look forward to introduce higher "moment" terms in Heavy-Ball optimisation Studies on Optimisation Algorithms in Machine Learning and look at how they affect the training process in terms of rate of convergence, probability of convergence, etc. We call our model the HEavier-Ball Optimisation (HerBo).

We look forward to make this project an exhaustive study by focussing on one or more benchmark problems (like MNIST, to compare with existing optimisation methods), include ablative tests on hyper-parameters (like learning rate, a, b) and parameters, and thus compare how our proposed change fares with existing methods.

The "MINST_via_CNN.ipynb" file has the results for the CNN architecture similar to that used in the Adam paper, we ran a plain plain SGD and achieved a 71.65% accuracy. This was for 5 epochs, training size of 2000 images and testing size of 2000 images. The results also have the testing accuracy after every epoch.

NOTE: The code in this repo is an improved version of what is present in https://github.com/CNN-NISER.
Until the final version comes out, if you feel the code in this repo is not understandable, see the other link for better understanding of what is going on, and then one can come back to see what we are trying to do here. 

## Recent Updates
- CNN with SGD for MNIST data was completed. (04/11/20)

- Adam optimiser code was added. (06/11/20)

- Included a feature that can run testing data to get test accuracy of the model once in every 'n' epochs. (06/11/20)

## Upcoming Upgrades
 - Obtain accuracy results for the Adam optimiser using a CNN for the MNIST data.
 
 - Include ability to run in mini-batches for SGD.

 - Code the proposed HerBo optimisation method in a CNN to tackle the MNIST database.
