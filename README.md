# DCASE2020 Task1
[Task1a](https://github.com/MihawkHu/DCASE2020_task1/tree/master/task1a) | [Task1b](https://github.com/MihawkHu/DCASE2020_task1/tree/master/task1b) | [Arxiv](https://arxiv.org/abs/2011.01447) | <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> | <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" />

**New** We add a [**list**](https://github.com/MihawkHu/DCASE2020_task1/blob/master/README.md#more-recent-related-works) on recent **related ASC works** containing discussion with this open resource ASC studies. Welcome to open an issue for adding related reference with the open resouce studies.   

This work has been accepted to IEEE ICASSP 2021! (Session Time: Friday, 11 June, 13:00 - 13:45 presented by Hu Hu)

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

If this work helps or has been used in your research, please consider to cite both papers below. Thank you!

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

### More Recent Related Works

Noted We simply generated the lists from [reference tools](https://scholar.google.com/scholar?cites=13758924980582482070&as_sdt=5,38&sciodt=0,38&hl=en). Feel free to pin us if you would like to share your work here. 

- Related to `Hu et al. "A Two-Stage Approach to Device-Robust Acoustic Scene Classification." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.`

| Title      |   Authors & Paper Link | Proc. |
| :---         |      :----:   |  :---: | 
| Attentive Max Feature Map for Acoustic Scene Classification with Joint Learning considering the Abstraction of Classes    | [Shim, Hye-jin, et al.](https://arxiv.org/pdf/2104.07213.pdf) | Arxiv 2021 |
|  Unsupervised Multi-Target Domain Adaptation for Acoustic Scene Classification  | [Dongchao Yang, et al.](https://arxiv.org/pdf/2105.10340v1.pdf)   | Arxiv 2021 |

- Related to `Hu, Hu, et al. "Device-robust acoustic scene classification based on two-stage categorization and data augmentation." DCASE (2020)`

| Title      |   Authors & Paper Link | Proc. |
| :---         |      :----:   |  :---: | 
| Acoustic scene classification in dcase 2020 challenge: generalization across devices and low complexity solutions    | [T Heittola, et al. ](https://arxiv.org/pdf/2005.14623.pdf) | DCASE 2020|
| CNN-Based Acoustic Scene Classification System    | [Y Lee t al.](https://www.mdpi.com/2079-9292/10/4/371/pdf) | Electronics 2021|
|Relational Teacher Student Learning with Neural Label Embedding for Device Adaptation in Acoustic Scene Classification| [Hu et al.](https://arxiv.org/pdf/2008.00110.pdf) | Arxiv 2020 |
|Attentive Max Feature Map for Acoustic Scene Classification with Joint Learning considering the Abstraction of Classes| [H Shim et al.](https://arxiv.org/pdf/2104.07213.pdf) | Arxiv 2021 |
|A Two-Stage Approach to Device-Robust Acoustic Scene Classification| [Hu et al.](https://ieeexplore.ieee.org/abstract/document/9414835) | ICASSP 2021 |
|Slow-Fast Auditory Streams for Audio Recognition| [E Kazakos et al.](https://ieeexplore.ieee.org/abstract/document/9413376/?casa_token=4NeKa18wFhgAAAAA:St-kJhc7IVINo6_OTrG1GzIFZfJqzdTDjsjNr4DSquSy0iha-sPNA4sGcq7x1376t4zWJ4z9Ma8) | ICASSP 2021|


## Acknowledgements
Codes borrows heavily from [DCASE2019-Task1](https://github.com/McDonnell-Lab/DCASE2019-Task1) and [dcase2020_task1_baseline](https://github.com/toni-heittola/dcase2020_task1_baseline). We appreciate the researchers contributing to this ASC community.


