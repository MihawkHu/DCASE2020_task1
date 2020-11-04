# DCASE2020 task 1a -- Acoustic Scene Classification with Multiple Devices

## Introduction

Task 1a focuses on ASC of audio signals recorded with multiple (real and simulated) devices into ten different fine-grained classes.   
 
In our submission([Technical Report](https://arxiv.org/abs/2007.08389)), we propose a novel two-stage ASC system leveraging upon ad-hoc score combination of two convolutional neural networks (CNNs), classifying the acoustic input according to three classes, and then ten classes, respectively. Different CNN-based architectures are explored to implement the two-stage classifiers, and several data augmentation techniques are also investigated. On Task 1a development data set, an ASC accuracy of 76.9\% is attained using our best single classifier and data augmentation. An accuracy of 81.9\% is then attained by a final model fusion of our two-stage ASC classifiers.

## Experimental results 
Tested on [DCASE 2020 task 1a development data set](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-a). The train-test split way follows the official recomendation.  

| System       |   Dev Acc. | 
| :---         |      :----:   | 
| Official Baseline     | 51.4%  | 
|  10-class FCNN  | 76.9%    | 
|  10-class Resnet  | 74.6%    | 
|  10-class fsFCNN  | 76.2%    | 
|  Two-stage ensemble system  |  81.9%   | 


## How to use

### Model training
To train 3-class FCNN, please run
> \$ cd 3class/fcnn    
> \$ python train_fcnn.py    

To train 3-class Resnet, please run
> \$ cd 3class/resnet  
> \$ python train_resnet.py  

To train 10-class FCNN, please run
> \$ cd 10class/fcnn   
> \$ python train_fcnn.py  

To train 10-class Resnet, please run
> \$ cd 10class/resnet   
> \$ python train_resnet.py  

To train 10-class fsFCNN, please run
> \$ cd 10class/fsFCNN  
> \$ python train_fsfcnn.py  

### Data augmentation 
We have totally 9 data augmentation methods used in task 1a:
1 Mixup.
2 Random cropping.
3 Spectrum augmentation.
4 Spectrum correction.
5 Reverberation with dynamic range compression.
6 Pitch shift. 
7 Speed change.
8 Random noise.
9 Mix audios.

Method 1, 2, 3 do not generate extra data so it's implementation in the training phase. Method 4 is in the folder `./data_augmentation/spectrum_correction`. Method 5 is in the folder `./data_augmentation/reverb_drc`. Method 6, 7, 8, 9 are in the folder `./data_augmentation/audio_based`. Please refer to the `README.md` file in each folder for usage instructions.


### Evaluation
To evaluate 3-class fusioned classifiers, please run
> \$ cd 3class  
> \$ python eval_fusion_vote_3class.py  

To evaluate 10-class single classifier, please run
> \$ cd 10class  
> \$ python eval_singlemodel.py  

To evaluate 10-class fusioned classifiers, please run
> \$ cd 10class  
> \$ python eval_fusion_vote_10class.py  


To evaluate final two-stage system (3-class + 10-class), please run 
> \$ python eval_hybrid.py

 

## Pre-trained models
Pre-trained models are provided in `./3class/pretrained_models` and `./10class/pretrained_models`, including
* FCNN for 3-class classification
* Resnet for 3-class classification
* FCNN for 10-class classification
* Resnet for 10-class classification
* fsFCNN for 10-class classification

Please note that due to the size limitation of github, we compressed and splitted the fsFCNN model. Please recover it before evaluation.

If you directly evaluate our provided pre-trained models by `eval_hybried.py`, you can get our reported results of 81.9% as follows, 

```shell
Val acc:  0.819
Val log loss: 0.936


Devices list:  ['a' 'b' 'c' 's1' 's2' 's3' 's4' 's5' 's6']
Per-device val acc :  [0.879 0.809 0.873 0.818 0.77  0.824 0.83  0.791 0.776]  
Device A acc:  0.879  
Device B & C acc:  0.841  
Device s1 & s2 & s3 acc:  0.804  
Device s4 & s5 & s6 acc:  0.799  
  
  
Confusion matrix:  
[[220   0   0  10   0   0  50  17   0   0]  
 [  0 280   6   0   0   0   0   0   0  11]    
 [  0  15 254   1   0   0   0   0   0  27]  
 [ 10   1   2 256   0   0  25   0   0   3]  
 [  0   0   0   0 282  10   0   1   4   0]  
 [  0   0   0   0  48 200   0  21  28   0]  
 [ 38   0   0  17   0   0 240   2   0   0]  
 [ 16   0   0   9   7  26  37 186  16   0]  
 [  0   0   0   0  14   8   0   5 270   0]  
 [  0  29  23   0   1   0   0   0   0 244]]  
Class names: ['airport' 'bus' 'metro' 'metro_station' 'park' 'public_square'
 'shopping_mall' 'street_pedestrian' 'street_traffic' 'tram']  
Per-class val acc:  [0.741 0.943 0.855 0.862 0.949 0.673 0.808 0.626 0.909 0.822  
```

## Conda Setup 

- Note: conda install would depend on your conda version. Please make sure you are using a latest conda version.

```bash
conda env create -f environment.yml
conda activate d20-keras
```

## FAQ

1. Is `data augmentation` important to reproduce the results in the `DCASE 2020 Task 1-a` evaluation set?

- Yes, we conducted several experiments and architecture-wise studies (refer to our technical report) and found out that it is easy to be overfitting with a major part of the evaluation subset. 

If you have challenges to access sufficient resources for training with large augmentation data, we also provide pre-trained models in this repo for future studies to the community.

