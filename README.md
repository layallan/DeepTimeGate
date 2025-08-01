# DeepTimeGate

DeepTimeGate is a PyTorch-based temporal segmentation model designed to process input scattering frames (`scat_frames`) and generate accurate mask predictions (`mask_frames`).

# Paper

# Title: Hybrid Deep Reconstruction for Vignetting-Free Upconversion Imaging through Scattering in ENZ Materials

Author: Hao Zhang*†, Yang Xu†, Wenwen Zhang, Saumya Choudhary, M. Zahirul Alam, Long D. Nguyen, Matthew Klein, Shivashankar Vangala, J. Keith Miller, Eric G. Johnson, Joshua R. Hendrickson, Robert W. Boyd and Sergio Carbajo


![Figure 1](image/Figure1.png)
## Files

- `train_deepgate.py` – training script  
- `test_deepgate.py` – testing script  
- `best_deeptimegate.pth` – pretrained model (download below)  
- `scat_frames/` – input images  
- `mask_frames/` – ground truth masks  

## Setup

```bash
pip install -r requirements.txt
```

## Dataset & Pretrained Model

Download from Google Drive:

- [Input (scat_frames)](https://drive.google.com/drive/folders/1eTaAwa4XMGXxxxYOSrMnFJLggv51AgU6?usp=sharing)  
- [Ground Truth (mask_frames)](https://drive.google.com/drive/folders/1eTaAwa4XMGXxxxYOSrMnFJLggv51AgU6?usp=sharing)  
- [Pretrained Model](https://drive.google.com/drive/folders/1eTaAwa4XMGXxxxYOSrMnFJLggv51AgU6?usp=sharing)

Place them as:

```
data/
├── scat_frames/
└── mask_frames/

models/
└── best_deeptimegate.pth
```

## Training

```bash
python train_deepgate.py --data_dir ./data --save_dir ./models
```

## Testing

```bash
python test_deepgate.py --data_dir ./data --model_path ./models/best_deeptimegate.pth --output_dir ./results
```

## Citation
Please cite:
https://doi.org/10.5281/zenodo.16551119
