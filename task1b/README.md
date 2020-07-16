## Task 1b Readme

### Tensorflow 2.2 compression

- Prerequisite

```bash
CUDA 10.2 (for tensorflow 2.2)
tensorflow 2.2
```

```bash
! pip uninstall -y tensorflow
! pip install -q tf-nightly
! pip install -q tensorflow-model-optimization
```

- Step Note

Step Notes our team member [YuanJun (Max) Zhao](zhaoyj1122).

```bash
eval.py       ## For testing the original keras model, which is saved as .hdf5 format. (This is an early version. Cropping is used to make sure the size of the features in training and evaluation sets is the same.)
model_trans.py ## For quantizing the keras model to TF lite model.
interpreter.py ## For testing the TF Lite model, which is saved as .tflite format.
```

STEP1: Put the trained keras model in the folder;

STEP2: Use eval.py to test the keras model on validation/evaluation set;

STEP3: Use model_trans.py to apply post-training quantization on the original keras model;

STEP4: Use interpreter.py to test the quantized model (.tflite file) on validation/evaluation set. Resultes can be saved in a csv file. 

### Colab compression

