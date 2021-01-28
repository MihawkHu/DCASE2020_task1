## Tensorflow >= 2.0 pretrained models

<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> 

This is a collection of pretrained models in TF >= 2.0 format.

You may easy check the pretrained models by running the commands below: 

- DCASE 2020 task 1-a (10 classes)

### FCNN-Attention

```shell
python fcnn_att.py
```

### ResNet

```shell
python resnet.py
```

- DCASE 2020 task 1-b (3 classes)

### Mobile V2

```shell
python mobnet.py 
```

Note: If this work helps or has been used in your research, please consider to cite the papers below. Thank you!

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
