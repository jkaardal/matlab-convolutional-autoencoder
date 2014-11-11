matlab-convolutional-autoencoder
================================

Cost function (cautoCost2.m) and cost gradient function (dcautoCost2.m) for a convolutional autoencoder.  

The network architecture is fairly limited, but these functions should be useful for unsupervised learning applications where input is convolved with a set of filters followed by reconstruction. It is also useful for discovering translationally invariant features of the data.  

Input is fed into the convolution layer, which is a set of filters applied to all user-defined subsets of the data. The input-output function of the convolution layer is a sigmoid. The reconstruction layer (or output layer) is linear. Optional extra hidden layers sandwiched between the convolution and reconstruction layer are sigmoid. Further information may be found in the comments in the file cautoCost2.m.

NOTE: This code uses parfor in a few places, which is a parallelized for loop. This requires the parallelization toolbox. If you do not have the parallelization toolbox, replace the parfor loops with for loops.
