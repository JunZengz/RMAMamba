# A Reverse Mamba Attention Network for Pathological Liver Segmentation

## Overview

## Create Environment
```
conda create -n RMAMamba python==3.8.16
conda activate RMAMamba
```

## Install Dependencies:
```    
pip install -r requirements.txt
cd selective_scan && pip install .
```

## Download Checkpoint 
Download pretrained checkpoints from [Google Drive](https://drive.google.com/drive/xxx) and move it to the `pretrained_pth` directory.

## Download Dataset
Download CirrMRI600+ dataset from [this link](https://osf.io/cuk24/) and move it to the `data` directory.

## Train
```
python train.py --model RMAMamba_S
```

## Test
```
python test.py --model RMAMamba_S
```

## Weight Files 
Our weight files and result maps are available on [Google Drive](https://drive.google.com/drive/xxx).


## Citation
Please cite our paper if you find the work useful:
```
@article{zeng2024rmamamba,
    title={A Reverse Mamba Attention Network for Pathological Liver Segmentation},
    author={Zeng, Jun and Jha, Debesh and Bagci, Ulas},
    bookarticle={xxx},
    year={2024},
}
```

## Contact
```
Please contact zeng.cqupt@gamil.com for any further questions.
```