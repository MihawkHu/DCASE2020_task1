# Data augmentation 

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

Method 1, 2, 3 do not generate extra data so it's implementation in the training phase. Method 4 is in the folder `spectrum_correction`. Method 5 is in the folder `reverb_drc`. Method 6, 7, 8, 9 are in the folder `audio_based`. 


## How to use 
Please refer to the `README.md` file in each folder for usage instructions. 