# Domain Generalization on Medical Imaging Classification Using Episodic Training with Task Augmentation (NCA 2021) ([Link](https://www.sciencedirect.com/science/article/abs/pii/S0010482521009380))

A Pytorch Implementation of ''Domain Generalization on Medical Imaging Classification Using Episodic Training with Task Augmentation'', which is accepted by the jounal of Computers in Biology and Medicine.

## Requirements

- Python == 3.7.4
- Tensorflow == 1.14.0
- CUDA 8.0


## Epithelium-stroma classification
### Train 
```
python main_mame.py
```

### Test
```
python test_mame.py
```

## Liver segmentation
### Train 
```
python main_seg_mame.py
```

### Test
```
python test_seg_mame.py
```

## Results on epithelium-stroma classification
|    Source    | Target |  MLDG  | Epi-FCR | MetaReg | JiGen |  MASF | Ours
|  ----------  |  ----  |  ----  | ------- | ------- | ----- |  ---- | ----
| NKI,IHC,NCH  |  VGH   |  91.13 |  91.49  |  91.74  | 92.05 | 92.43 | 93.51

## Results on liver segmentation
|      Source      | Target |  MLDG  | Epi-FCR | MetaReg | JiGen |  MASF | Ours
|  --------------  |  ----  |  ----  | ------- | ------- | ----- |  ---- | ----
| BTCV,CHAOS,LITS  | IRCAD  |  89.17 |  89.26  |  89.17  | 91.44 | 90.89 | 92.14


## Citation
If you find this repository useful, please cite our paper:
```
@article{li2022domain,
  title={Domain generalization on medical imaging classification using episodic training with task augmentation},
  author={Li, Chenxin and Lin, Xin and Mao, Yijin and Lin, Wei and Qi, Qi and Ding, Xinghao and Huang, Yue and Liang, Dong and Yu, Yizhou},
  journal={Computers in Biology and Medicine},
  volume={141},
  pages={105144},
  year={2022},
  publisher={Elsevier}
}
```
