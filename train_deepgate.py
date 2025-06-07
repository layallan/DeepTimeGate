import os
import glob
import shutil
import random
import multiprocessing
import time
import logging
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


SCAT_ROOT         = 'scat_frames'
MASK_ROOT         = 'mask_frames'
SPLIT_ROOT        = 'dataset_all'
RANDOM_SEED       = 42

BATCH_SIZE        = 8
NUM_EPOCHS        = 2000
# NUM_EPOCHS        = 3
PATIENCE          = 60
LEARNING_RATE     = 1e-6
IMAGE_SIZE        = (256, 256)

NUM_WORKERS_TRAIN = 4
NUM_WORKERS_VAL   = 2
NUM_WORKERS_TEST  = 2

LOG_FILE          = 'training_deepgate.log'
BEST_MODEL_PATH   = 'best_deeptimegate.pth'
LOSS_CURVE_PATH   = 'loss_curve_deepgate.pdf'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE)
ch = logging.StreamHandler()
fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(fmt); ch.setFormatter(fmt)
logger.addHandler(fh); logger.addHandler(ch)

class ScatMaskDataset(Dataset):
    def __init__(self, root_dir, split,
                 geom_transform=None, img_transform=None, mask_transform=None):
        self.scat_dir = os.path.join(root_dir, split, 'scat')
        self.mask_dir = os.path.join(root_dir, split, 'mask')
        self.scat_paths = sorted(glob.glob(os.path.join(self.scat_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        assert len(self.scat_paths)==len(self.mask_paths), "Mismatch scat/mask counts"
        self.geom_transform = geom_transform
        self.img_transform  = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.scat_paths)

    def __getitem__(self, idx):
        scat_path = self.scat_paths[idx]
        mask_path = self.mask_paths[idx]
        stem = os.path.basename(scat_path)
        scat = Image.open(scat_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        if self.geom_transform:
            seed = random.randint(0,2**32)
            random.seed(seed); scat = self.geom_transform(scat)
            random.seed(seed); mask = self.geom_transform(mask)
        if self.img_transform:  scat = self.img_transform(scat)
        if self.mask_transform: mask = self.mask_transform(mask)
        return scat, mask, stem


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512, 1024]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        for feat in features:
            self.downs.append(DoubleConv(in_ch, feat))
            in_ch = feat
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, 2, 2))
            self.ups.append(DoubleConv(feat*2, feat))
        self.pool  = nn.MaxPool2d(2,2)
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips=[]
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)
        return self.final(x)

class DIPModule(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64,    64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64,    out_ch,3, padding=1)
        )
    def forward(self, x):
        return self.net(x)


def compute_iou(pred, target, thresh=0.5):
    pred_bin = (pred>thresh).float()
    inter    = (pred_bin*target).sum()
    union    = pred_bin.sum()+target.sum()-inter
    return (inter+1e-6)/(union+1e-6)


def main():
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True
    random.seed(RANDOM_SEED)


    if not os.path.isdir(SCAT_ROOT) or not os.path.isdir(MASK_ROOT):
        logger.error("Missing SCAT_ROOT or MASK_ROOT")
        return
    paired=[]
    for s in sorted(glob.glob(os.path.join(SCAT_ROOT,'*','*.png'))):
        rel = os.path.relpath(s,SCAT_ROOT)
        fld,fn = os.path.split(rel)
        m = os.path.join(MASK_ROOT,fld,fn.replace('scat_','mask_'))
        if os.path.exists(m):
            paired.append((s,m))
    random.shuffle(paired)
    n,i1,i2 = len(paired), int(0.8*len(paired)), int(0.9*len(paired))
    subsets = {'train':paired[:i1], 'val':paired[i1:i2], 'test':paired[i2:]}
    for sp,ps in subsets.items():
        for d in ['scat','mask']:
            os.makedirs(os.path.join(SPLIT_ROOT,sp,d), exist_ok=True)
        for s,m in ps:
            fn=os.path.basename(s)
            shutil.copy(s, os.path.join(SPLIT_ROOT,sp,'scat',fn))
            shutil.copy(m, os.path.join(SPLIT_ROOT,sp,'mask',fn))
    logger.info(f"Data split: train={len(subsets['train'])}, val={len(subsets['val'])}, test={len(subsets['test'])}")


    geom     = transforms.Resize(IMAGE_SIZE)
    to_tensor= transforms.ToTensor()
    train_dl = DataLoader(ScatMaskDataset(SPLIT_ROOT,'train',geom,to_tensor,to_tensor),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS_TRAIN, pin_memory=True)
    val_dl   = DataLoader(ScatMaskDataset(SPLIT_ROOT,'val',  geom,to_tensor,to_tensor),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS_VAL,   pin_memory=True)
    test_dl  = DataLoader(ScatMaskDataset(SPLIT_ROOT,'test', geom,to_tensor,to_tensor),
                          batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS_TEST,  pin_memory=True)


    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet      = UNet().to(device)

    dip_mod   = DIPModule().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        list(unet.parameters()) + list(dip_mod.parameters()),
        lr=LEARNING_RATE)


    best_val, no_imp = float('inf'), 0
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS+1):
        unet.train(); dip_mod.train()
        tot_train=0.0
        for x,y,_ in train_dl:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            p_low = unet(x)
            res   = dip_mod(p_low)
            hr    = torch.clamp(p_low + res,0,1)
            loss  = criterion(p_low,y) + criterion(hr,y)
            loss.backward(); optimizer.step()
            tot_train += loss.item() * x.size(0)
        train_loss = tot_train / len(train_dl.dataset)
        train_losses.append(train_loss)

        unet.eval(); dip_mod.eval()
        tot_val=0.0
        with torch.no_grad():
            for x,y,_ in val_dl:
                x,y = x.to(device), y.to(device)
                p_low = unet(x)
                hr    = torch.clamp(p_low + dip_mod(p_low),0,1)
                lval  = (criterion(p_low,y) + criterion(hr,y)).item()
                tot_val += lval * x.size(0)
        val_loss = tot_val / len(val_dl.dataset)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch}/{NUM_EPOCHS}  Train {train_loss:.6f}  Val {val_loss:.6f}")
        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
            torch.save({'unet':unet.state_dict(),'dip':dip_mod.state_dict()}, BEST_MODEL_PATH)
            logger.info(" Saved best model")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    logger.info(f"Training completed in {time.time()-start_time:.1f}s")


    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)


    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    unet.load_state_dict(ckpt['unet'])
    dip_mod.load_state_dict(ckpt['dip'])
    unet.eval(); dip_mod.eval()

    os.makedirs('test_results', exist_ok=True)
    metrics = {'scatter':{'psnr':[],'ssim':[],'iou':[]},
               'unet':   {'psnr':[],'ssim':[],'iou':[]},
               'dtg':    {'psnr':[],'ssim':[],'iou':[]}}
    records=[]

    def to_pil(tensor):
        arr = (tensor.cpu().squeeze().numpy()*255).astype('uint8')
        return Image.fromarray(arr).convert('L')

    for scat,mask,stem in test_dl:
        scat,mask = scat.to(device), mask.to(device)
        stem = stem[0]
        mask_np = mask.cpu().squeeze().numpy()
        scat_np = scat.cpu().squeeze().numpy()


        m_psnr = peak_signal_noise_ratio(mask_np,scat_np,data_range=1.0)
        m_ssim = structural_similarity(mask_np,scat_np,data_range=1.0)
        m_iou  = compute_iou(scat.cpu().squeeze(),mask.cpu().squeeze())
        metrics['scatter']['psnr'].append(m_psnr)
        metrics['scatter']['ssim'].append(m_ssim)
        metrics['scatter']['iou'].append(m_iou)


        with torch.no_grad():
            p_low = unet(scat)
        low_np = p_low.cpu().squeeze().numpy()
        u_psnr = peak_signal_noise_ratio(mask_np,low_np,data_range=1.0)
        u_ssim = structural_similarity(mask_np,low_np,data_range=1.0)
        u_iou  = compute_iou(p_low.cpu().squeeze(),mask.cpu().squeeze())
        metrics['unet']['psnr'].append(u_psnr)
        metrics['unet']['ssim'].append(u_ssim)
        metrics['unet']['iou'].append(u_iou)

        with torch.no_grad():
            res = dip_mod(p_low)
            hr  = torch.clamp(p_low+res,0,1)
        hr_np = hr.cpu().squeeze().numpy()
        d_psnr = peak_signal_noise_ratio(mask_np,hr_np,data_range=1.0)
        d_ssim = structural_similarity(mask_np,hr_np,data_range=1.0)
        d_iou  = compute_iou(hr.cpu().squeeze(),mask.cpu().squeeze())
        metrics['dtg']['psnr'].append(d_psnr)
        metrics['dtg']['ssim'].append(d_ssim)
        metrics['dtg']['iou'].append(d_iou)


        orig_img = to_pil(scat); gt_img = to_pil(mask)
        unet_img = to_pil(p_low); dtg_img = to_pil(hr)
        W,H = orig_img.size
        comp = Image.new('L',(W*4,H))
        for i,im in enumerate([orig_img,gt_img,unet_img,dtg_img]):
            comp.paste(im,(i*W,0))
        comp.save(os.path.join('test_results',f"cmp_{stem}.png"))


        for method, psnr, ssim, iou in [
            ('scatter',m_psnr,m_ssim,m_iou),
            ('unet',   u_psnr,u_ssim,u_iou),
            ('dtg',    d_psnr,d_ssim,d_iou)
        ]:
            records.append({
                'sample': stem,
                'method': method,
                'psnr':   psnr,
                'ssim':   ssim,
                'iou':    iou
            })


    pd.DataFrame.from_records(records).to_csv('test_results/metrics.csv', index=False)


    for m in metrics:
        logger.info(f"{m.upper()} Mean PSNR: {np.mean(metrics[m]['psnr']):.3f}, "
                    f"SSIM: {np.mean(metrics[m]['ssim']):.3f}, "
                    f"IoU: {np.mean(metrics[m]['iou']):.3f}")

if __name__=='__main__':
    main()
