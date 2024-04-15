# MG-Transformer
<p align="center">
  <img src="images/MG-Transformer.png" alt="MG-Transformer" width="800"/>
</p>

## Installation and Dependencies
Create the `m2` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate m2
```
## Feature Extraction
Extract Region features with `./feature_pro/pre_region_feature.py` < br >
CLIP image embedding: `./feature_pro/pre_CLIP_feature.py` < br >
Group mask matrix: `./feature_pro/split_group.py`


## Train
```
python train.py
```

## Evaluate
```
python test.py
```


## Reference:
1. https://github.com/tylin/coco-caption
2. https://github.com/aimagelab/meshed-memory-transformer
3. https://github.com/One-paper-luck/PKG-Transformer
