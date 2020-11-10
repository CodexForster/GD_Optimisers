# Introducing Higher Moment Terms in Heavy-Ball Optimisation
In this project, we look forward to introduce higher "moment" terms in Heavy-Ball optimisation Studies on Optimisation Algorithms in Machine Learning and look at how they affect the training process in terms of rate of convergence, probability of convergence, etc. We call our model the HEavier-Ball Optimisation (HerBo).

We look forward to make this project an exhaustive study by focussing on one or more benchmark problems (like MNIST, to compare with existing optimisation methods), include ablative tests on hyper-parameters (like learning rate, a, b) and parameters, and thus compare how our proposed change fares with existing methods.

The "MINST_via_CNN.ipynb" file has the results for the CNN architecture similar to that used in the Adam paper, we multiple algorithms like Stochastic Gradient Descent (SGD), Mini-Batch Gradient Descent (MBGD), HerBo (for MBGD) and Adam (for MBGD and SGD) for the MNIST problem. This was for mulitple epochs, training size of 2560 images and testing size of 2560 images. The results also have the testing accuracy after every epoch. We see that rate of convergence for Adam MBGD > MBGD > HerBo MBGD > Plain SGD. All accuracy results are in the file. Please NOTE that there is an error that comes in the SGD part after the 8th epoch. We think that possibly occurred due to some parameter shooting to a large number, so for SGD, we have results only until 8 epochs.

"comparision.py" plots the accuracy scores after training for certain number of epochs. This essentially summarises our results and the comparision of all algorithms we have tested with.

NOTE: The code in this repo is an improved version of what is present in https://github.com/CNN-NISER.
Until the final version comes out, if you feel the code in this repo is not understandable, see the other link for better understanding of what is going on, and then one can come back to see what we are trying to do here. 

## Recent Updates

- Ran the same experiment on all for 10 epochs to obtain a better understanding of performance over epochs. (10/11/20)

- Code the proposed HerBo optimisation method in a CNN to tackle the MNIST database. (08/11/20)

- Obtained accuracy results for the Adam optimiser using a CNN for the MNIST data. Adam (SGD) and Adam using MBGD were run (mini-batch size of 128). (07/11/20)

- Included a feature that can run testing data to get test accuracy of the model once in every 'n' epochs. (06/11/20)

- Adam optimiser code was added. (06/11/20)

- CNN with SGD for MNIST data was completed. (04/11/20)

## Upcoming Upgrades
 
 - Run the same experiments for different MNIST training images, and for a larger dataset.
 
 - Use a closer CNN architecture to what was used in the Adam paper and perform all of the above tests to obtain acceptable results.
 
