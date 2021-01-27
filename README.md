# DCASE2020 Task1
[Task1a](https://github.com/MihawkHu/DCASE2020_task1/tree/master/task1a) | [Task1b](https://github.com/MihawkHu/DCASE2020_task1/tree/master/task1b) | [Arxiv](https://arxiv.org/abs/2011.01447)


## Introduction
This is an implementation of [DCASE 2020 Task 1a](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-a) and [DCASE 2020 Task 1b](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-b) on **Acoustic Scene Classification with Multiple Devices**. We attain 2nds for both Task-1a and Task-1b in the official challenge 2020.  [Technical Report](https://arxiv.org/abs/2007.08389).

We sincerely thank all the team members and advisors from [Georgia Tech ECE](https://chl.ece.gatech.edu/), [Tencent Media Lab](https://avlab.qq.com/#/index), [USTC](http://staff.ustc.edu.cn/~jundu/), and [Univeristy of Enna](https://www.unikore.it/index.php/it/ingegneria-informatica-persone/docenti-del-corso/itemlist/category/1589-siniscalchi).


## Experimental results
### Task 1a
Tested on [DCASE 2020 task 1a development data set](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-a). The train-test split way follows the official recomendation.  

| System       |   Dev Acc. | 
| :---         |      :----:   | 
| Official Baseline     | 51.4%  | 
|  10-class FCNN  | 76.9%    | 
|  10-class Resnet  | 74.6%    | 
|  10-class fsFCNN  | 76.2%    | 
|  Two-stage ensemble system  |  81.9%   | 


### Task 1b
Tested on [DCASE 2020 task 1b development data set](http://dcase.community/challenge2020/task-acoustic-scene-classification#subtask-b). The train-test split way follows the official recomendation.  

| System       |   Dev Acc. (size)<br> Original model| Dev Acc. (size) <br> Quantization | 
| :---         |      :----:   | :---: | 
| Official Baseline     | 87.3% (450K)   |  - | 
|   Mobnet  | 95.2% (3.2M)    | 94.8% (411K) | 
|   small-FCNN    |  96.4% (2.8M)    | 96.3% (357K) | 
|   Mobnet + small-FCNN-v1   | 96.8% (1.8M+1.9M)      | 96.7% (497K) | 
|   small-FCNN-v1 + small-FCNN-v2   | 96.5% (1.9M+2.1M)     | 96.3% (499K)| 


## How to use

### Task 1a
Please refer to the `README.md` in `./task1a/` for detailed instructions.

### Task 1b
Please refer to the `README.md` in `./task1b/` for detailed instructions.

### Pre-trained models
- Pre-trained keras models are provided in `./task1a/3class/pretrained_models/`, `task1a/10class/pretrained_models/`, and `./task1b/pretrained_models/`. You can get reported results by directly evaluate pre-trained models.

- tensorflow >= 2.0 pretrained models. We also provide some pretrained DCASE task1 models in tensorflow >= 2.0 format. 
Please refer to [`./other_TF2_format_pretrained/`](https://github.com/MihawkHu/DCASE2020_task1/tree/master/other_TF2_format_pretrained)

## Reference

If this work helps or has been used in your research, please consider to cite the paper below. Thank you!

```bib
@article{hu2020two,
  title={A Two-Stage Approach to Device-Robust Acoustic Scene Classification},
  author={Hu, Hu and Yang, Chao-Han Huck and Xia, Xianjun and Bai, Xue and Tang, Xin and Wang, Yajian and Niu, Shutong and Chai, Li and Li, Juanjuan and Zhu, Hongning and others},
  journal={arXiv preprint arXiv:2011.01447},
  year={2020}
}

@misc{hu2020devicerobust,
    title={Device-Robust Acoustic Scene Classification Based on Two-Stage Categorization and Data Augmentation},
    author={Hu Hu and Chao-Han Huck Yang and Xianjun Xia and Xue Bai and Xin Tang and Yajian Wang and Shutong Niu and Li Chai and Juanjuan Li and Hongning Zhu and Feng Bao and Yuanjun Zhao and Sabato Marco Siniscalchi and Yannan Wang and Jun Du and Chin-Hui Lee},
    year={2020},
    eprint={2007.08389},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

## Acknowledgements
Codes borrows heavily from [DCASE2019-Task1](https://github.com/McDonnell-Lab/DCASE2019-Task1) and [dcase2020_task1_baseline](https://github.com/toni-heittola/dcase2020_task1_baseline). We appreciate the researchers contributing to this ASC community.
