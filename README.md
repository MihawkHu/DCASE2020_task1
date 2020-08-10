# DCASE2020_task1 (Under building, Plan to release in August 17th 2020)

Note - we plan to have a major release in Aug. 17th 2020. Thank you for your interests. 

Under building.

This is an implementation of DCASE 2020 **Task 1a** and **Task 1b** on **Acoustic Scene Classification with Multiple Devices**. We attain 2nds for both Task-1a and Task-1b in the official challenge 2020. [Technical Report](https://arxiv.org/abs/2007.08389).

We sincerely thank all the team members and advisors from [Georgia Tech ECE](https://chl.ece.gatech.edu/), [Tencent Media Lab](https://avlab.qq.com/#/index), [USTC](http://staff.ustc.edu.cn/~jundu/), and [Univeristy of Enna](https://www.unikore.it/index.php/it/ingegneria-informatica-persone/docenti-del-corso/itemlist/category/1589-siniscalchi).

## Task 1a


| System      | Description | Dev Acc.     | Eval Acc. |
| :---        |    :----:   |      :----:   |         ---: |
| Baseline      |  CNNs     | 54.1%   | 51.4%|
|   Two-stage acoustic scene classification (ours) | Ensemble        | 81.9%      | 76.2%|


```bash
git clone https://github.com/MihawkHu/DCASE2020_task1/

cd DCASE2020_task1/task1a
```

## Task 1b

```bash
git clone https://github.com/MihawkHu/DCASE2020_task1/`

cd DCASE2020_task1/task1b
```

## Reference

If this work helps or has been used in your research, please consider to cite the paper below. Thank you!

https://arxiv.org/abs/2007.08389

```bib
@misc{hu2020devicerobust,
    title={Device-Robust Acoustic Scene Classification Based on Two-Stage Categorization and Data Augmentation},
    author={Hu Hu and Chao-Han Huck Yang and Xianjun Xia and Xue Bai and Xin Tang and Yajian Wang and Shutong Niu and Li Chai and Juanjuan Li and Hongning Zhu and Feng Bao and Yuanjun Zhao and Sabato Marco Siniscalchi and Yannan Wang and Jun Du and Chin-Hui Lee},
    year={2020},
    eprint={2007.08389},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
