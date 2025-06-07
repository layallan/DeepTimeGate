# DeepTimeGate

DeepTimeGate is a PyTorch-based temporal segmentation model designed to process input scattering frames (`scat_frames`) and generate accurate mask predictions (`mask_frames`).

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

- [Input (scat_frames)](YOUR_LINK_HERE)  
- [Ground Truth (mask_frames)](YOUR_LINK_HERE)  
- [Pretrained Model](YOUR_LINK_HERE)

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
