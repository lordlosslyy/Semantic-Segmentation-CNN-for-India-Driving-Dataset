/*
 * Filename: README
 * Authors: Zi-Xiang Xia, Yun-Yi Lin, Zhuojun Chen, Peiyi Li
 * Date: Feb 15, 2021
 */

----------------- [ DESCRIPTION ] -----------------
This program will build and evaluate basic FCN, improved FCN, Mobile-Deeplabv3+, VGG and U-Net for semantic segmentation on India Driving Dataset using Python and PyTorch.


----------------- [ HOW TO RUN ] -----------------
To run the baseline model - run 'python3 starter.py'

To run the improved baseline model by data augmentation, run 'python3 starter_4a.py' 

To run the improved baseline model by weighted loss, 'python3 weightedLoss.py'

To run the improved baseline model by dice loss, change "criterion = nn.CrossEntropyLoss(weight=normedWeights.cuda()).cuda()" to "criterion = DiceLoss().cuda()" and replace all "criterion(outputs, Y)" to " criterion(outputs, labels)" in line 71 and simply run 'python3 weightedLoss.py'

To run the Mobile-Deeplabv3+ - add "from deeplabv3_plus import *" to the top of the starter_weightedLoss.py; change "fcn_model = FCN(n_class=27)" to "fcn_model = DeepLab()" in line 72 and delete "fcn_model.apply(init_weights)" in line 73 and simply run 'python3 starter_weightedLoss.py'.

To run the transfer learning - vgg model with weighted loss, add "from transfer import *" to the top of the starter_weightedLoss.py; change "fcn_model = FCN(n_class=27)" to "fcn_model = Transfer(n_class=27)" in line 72 and delete "fcn_model.apply(init_weights)" in line 73 and simply run 'python3 starter_weightedLoss.py'.

To run the unet model with weighted loss, add "from unet import *" to the top of the starter_weightedLoss.py; change "fcn_model = FCN(n_class=27)" to "fcn_model = UNet(n_class=27)" in line 72 and simply run 'python3 starter_weightedLoss.py'.


----------------- [ OUTPUT ] -----------------
The program will load the data, build the neural networks, train and evaluate the model, and output the training and validation loss and accuracy plots, visualize the segmented outputs, and print the test pixel accuracy and IoU.


----------------- [ TESTING ] -----------------
We have tested the implementations of metrics, models, training process by a comprehensive suite of tests. 