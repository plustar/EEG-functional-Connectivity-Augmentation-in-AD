# EEG-functional-Connectivity-Augmentation-in-AD
The code repository for the paper "Assessing the potential of data augmentation in EEG functional connectivity for early detection of Alzheimer's disease". 

Code description:
Step1-A_read_origindata
Combine the separate data files into one file considering the processing in the following steps;
Step2-B_mode_decomposition
Decompose the EEG time series into multiple modes with classical/serial/multivariate mode decomposition.
Step3-C_generate_artifact
Split the dataset into the training set and the testing set, and generate the artifical data with the training set
Step4-D_CoherentObject
Calculate the functional connectivity with the EEG signals in the training/testing set
Step5_E_classification_brainnet/resnet/eegnet
Evaluate the model performance on the augmented dataset, including BrainNet CNN, ResNet-18, and EEGNet
Step6_F_collect_accur/confusion
Collect and calculate the classification performance of three models

The code for multivariate mode decomposition is from http://freesourcecode.net/matlabprojects/5896/Multivariate-Empirical-Mode-Decomposition-Matlab-Code.
