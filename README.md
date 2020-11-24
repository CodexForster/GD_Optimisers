# Introducing Higher Moment Terms in Heavy-Ball Optimisation
In this project, we look forward to introduce higher "moment" terms in Heavy-Ball optimisation Studies on Optimisation Algorithms in Machine Learning and look at how they affect the training process in terms of rate of convergence, probability of convergence, etc. We call our model the HEavier-Ball Optimisation (HerBo).

We look forward to make this project an exhaustive study by focussing on one or more benchmark problems (like MNIST, to compare with existing optimisation methods), include ablative tests on hyper-parameters (like learning rate a, b in adam and HerBo) and parameters, and thus compare how our proposed change fares with existing methods. Note that all HerBo tests were conducted for n=3.

For our first step, we ran for a sample test and train test. The "MINST_via_CNN_old_train_test.ipynb" is the sample file of how our output will look like for the tests we conduct. It has the results for the CNN architecture similar to that used in the Adam paper, we multiple algorithms like Stochastic Gradient Descent (SGD), Mini-Batch Gradient Descent (MBGD), HerBo (for MBGD) and Adam (for MBGD and SGD) for the MNIST problem. This was for 10 epochs, training size of 2560 images and testing size of 2560 images. The results also have the testing accuracy after every epoch. We see that rate of convergence for Adam MBGD > MBGD > HerBo MBGD > Plain SGD. All accuracy results are in the file. Please NOTE that there is an error that comes in the SGD part after the 8th epoch. We think that possibly occurred due to some parameter shooting to a large number, so for SGD, we have results only until 8 epochs. "comparision.py" plots the accuracy scores after training for certain number of epochs of the above stated models. This essentially summarises our results and the comparision of all algorithms we have tested with.

For the second step, we tried to have a good training dataset ("MINST_via_CNN_shrinked_train.ipynb"). Since it is not feasible (computationally) for us to train for all 60,000 MNIST images, we wanted to make the training set consisit of 2560 images (batch size is 128, so training size has to be a multiple of that). So we calculated the percentage of examples in each class in the original MNIST training set, and maintainted the same values in our smaller training dataset. Eg: 11.24% of the original MNIST dataset were 1s. So 11.24% of 2560 images of our training set are also 1s. We selected 260 images from the MNIST testing dataset as our testing dataset. And we ran this for all the above stated algorithms. We have uploaded the codes for the same in files named as "MINST_via_CNN_shrinked_train.ipynb", it can run any of these algorithms and also has the code for making our training and testing datasets. the results for this step are in "comparision2.py"

For the third step, we used the shrinked training dataset, but added the ability to perform cross validation. Each cross-validation fold had the training process go for 10 epochs. We conducted 5-fold cross-validation experiments for different values of the hyperparameter 'b' in the HerBo algorithm. Training dataset size was 2560 images and the testing dataset size was 640 images. Results and codes (in .ipynb files) for these configurations are in the "HerBo Outputs" folder. There are some files with part1 and part2 in their name, these are 2 parts of the same file. If the file has 2 parts, part1 was the file that was running and abruptly got ended due to long runtime. We had to run the rest again which is in part2. Accuracy plots of all configurations are in "comparision3.py".

NOTE: The code in this repo is an improved version of what is present in https://github.com/CNN-NISER.
Until the final version comes out, if you feel the code in this repo is not understandable, see the other link for better understanding of what is going on, and then one can come back to see what we are trying to do here. 

## Recent Updates

- Run tests for HerBo with 5-fold cross-validation using different hyper-parameters to see what values suit it. (23/11/20)

- Run the same experiments for different MNIST training images, and for a more inclusive dataset. (12/11/20)

- Ran the same experiment on all for 10 epochs to obtain a better understanding of performance over epochs. (10/11/20)

- Code the proposed HerBo optimisation method in a CNN to tackle the MNIST database. (08/11/20)

- Obtained accuracy results for the Adam optimiser using a CNN for the MNIST data. Adam (SGD) and Adam using MBGD were run (mini-batch size of 128). (07/11/20)

- Included a feature that can run testing data to get test accuracy of the model once in every 'n' epochs. (06/11/20)

- Adam optimiser code was added. (06/11/20)

- CNN with SGD for MNIST data was completed. (04/11/20)

## Upcoming Upgrades
 
 - Use a closer CNN architecture to what was used in the Adam paper and perform all of the above tests to obtain acceptable results.
