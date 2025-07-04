# A Reverse Mamba Attention Network for Pathological Liver Segmentation

## Overview

## Create Environment
```
conda create -n RMAMamba python==3.8.16
conda activate RMAMamba
```

## Install Dependencies
```    
pip install -r requirements.txt
cd selective_scan && pip install .
```

## Download Checkpoint 
Download pretrained checkpoints from [Google Drive](https://drive.google.com/file/d/1fsGNq_0ZwHtjrPAuAen2iIkPaRW9ppx4/view?usp=sharing) and move it to the `pretrained_pth` directory.

## Download Dataset
Download CirrMRI600+ dataset from [this link](https://osf.io/cuk24/) or [Google Drive](https://drive.google.com/file/d/1JPbsYEfPgqZEh-Y2wJtqwLp63ix3NqEv/view?usp=drive_link).
Move it to the `data` directory.

## Train
```
python train.py --model RMAMamba_S
```

## Test
```
python test.py --model RMAMamba_S
```

## Weight Files 
Our weight files and result maps are available on [Google Drive](https://drive.google.com/file/d/1rQw6EE2zUTstVxhXPk8FGLb3kFTHdDd2/view?usp=drive_link).


## Citation
Please cite our paper if you find the work useful:
```
@article{zeng2025reverse,
  title={A Reverse Mamba Attention Network for Pathological Liver Segmentation},
  author={Zeng, Jun and Jha, Debesh and Aktas, Ertugrul and Keles, Elif and Medetalibeyoglu, Alpay and Antalek, Matthew and Lewandowski, Robert and Ladner, Daniela and Borhani, Amir A and Durak, Gorkem and others},
  journal={arXiv preprint arXiv:2502.18232},
  year={2025}
}
```

## Contact

Please contact zeng.cqupt@gamil.com for any further questions.
