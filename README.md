# Domain-generalization-with-task-augmentation-for-MIA
## Requirements

- Python == 3.7.4
- Tensorflow == 1.14.0
- CUDA 8.0


## Epithelium-stroma classification
### Train 
* python `main_mame.py`

### Test
* python `test_mame.py`

## Liver segmentation
### Train 
* python `main_seg_mame.py`

### Test
* python `test_seg_mame.py`

## Results on epithelium-stroma classification
|    Source    | Target |  MLDG  | Epi-FCR | MetaReg | JiGen |  MASF | Ours
|  ----------  |  ----  |  ----  | ------- | ------- | ----- |  ---- | ----
| NKI,IHC,NCH  |  VGH   |  91.13 |  91.49  |  91.74  | 92.05 | 92.43 | 93.51

## Results on liver segmentation
|      Source      | Target |  MLDG  | Epi-FCR | MetaReg | JiGen |  MASF | Ours
|  --------------  |  ----  |  ----  | ------- | ------- | ----- |  ---- | ----
| BTCV,CHAOS,LITS  | IRCAD  |  89.17 |  89.26  |  89.17  | 91.44 | 90.89 | 92.14

See more details in the paper
