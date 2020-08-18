# DCASE2020 task 1b -- Low-Complexity Acoustic Scene Classification

## Introduction
In Task 1b, the main goal is to keep the system size within **500 Kilobytes (KB)**. 

In our submission ([Technical Report](https://arxiv.org/abs/2007.08389)), a post-training quantization method, which is provided by [Tensorflow2](https://www.tensorflow.org/tutorials), is used to compress our neural models. We used **Dynamic Range Quantization (DRQ)**, in which neural weights are quantized from floating-point to integer having a 8-bit precision. Quantization not only reduces the model size but also improves hardware accelerator latency with little degradation in final classification accuracy. Leveraging DRQ, we thus transferred our neural architectures from a 32-bit TensorFlow format to a 8-bit TensorFlow-lite format, which compresses the model size to about 1/8 of its original size. According to our experimental evidence, such a compression method resulted in a minor ASC classification drop.


## Experimental results 
Tested on [DCASE 2020 task 1b development data set](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-b). The train-test split way follows the official recomendation.  

| System       |   Dev Acc. (size)<br> Original model| Dev Acc. (size) <br> Quantization | 
| :---         |      :----:   | :---: | 
| Official Baseline     | 87.3% (450K)   |  - | 
|   Mobnet  | 95.2% (3.2M)    | 94.8% (411K) | 
|   small-FCNN    |  96.4% (2.8M)    | 96.3% (357K) | 
|   Mobnet + small-FCNN-v1   | 96.8% (1.8M+1.9M)      | 96.7% (497K) | 
|   small-FCNN-v1 + small-FCNN-v2   | 96.5% (1.9M+2.1M)     | 96.3% (499K)| 


## How to use

### Model training
To train Mobnet, please run
> \$ cd train  
> \$ python train_mobnet.py  

To train small-FCNN, please run
> \$ cd train  
> \$ python train_smallfcnn.py  

### Quantization
To compress well-trained model by quantization, please run
> \$ cd quantization  
> \$ python model_trans.py  

### Evaluation
To evaluate trained models and quantized models, please run
> \$ cd eval  
> \$ python eval_original.py  \# evaluate original model  
> \$ python eval_quantized.py  \# evaluate quantized model  
> \$ python eval_quantized_fusion_LR.py  \# evaluate two fusioned models by logistic regression strategy  
> \$ python eval_quantized_fusion_AVG.py  \# evaluate two fusioned models by average strategy   
 

## Pre-trained models
Pre-trained models are provided in `./pretrained_models`, including
* Mobnet (before and after quantization)
* small-FCNN (before and after quantization)
 