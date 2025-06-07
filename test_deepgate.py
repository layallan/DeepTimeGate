
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


SPLIT_ROOT        = 'dataset_all'
BEST_MODEL_PATH   = './deepgate/best_deeptimegate.pth'
TEST_RESULTS_DIR  = 'test_results_deepgate'
BATCH_SIZE        = 1
IMAGE_SIZE        = (256, 256)
NUM_WORKERS_TEST  = 2


from train_deepgate import UNet, DIPModule, ScatMaskDataset, compute_iou

def to_pil(tensor):
    arr = (tensor.cpu().squeeze().numpy() * 255).astype('uint8')
    return Image.fromarray(arr).convert('L')


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geom     = transforms.Resize(IMAGE_SIZE)
    to_tensor = transforms.ToTensor()
    test_dataset = ScatMaskDataset(SPLIT_ROOT, 'test', geom, to_tensor, to_tensor)
    test_dl = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS_TEST,
                         pin_memory=True)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    unet      = UNet().to(device)
    dip_mod   = DIPModule().to(device)
    unet.load_state_dict(checkpoint['unet'])
    dip_mod.load_state_dict(checkpoint['dip'])
    unet.eval(); dip_mod.eval()

    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    metrics = {'scatter': {'psnr': [], 'ssim': [], 'iou': []},
               'unet':    {'psnr': [], 'ssim': [], 'iou': []},
               'dtg':     {'psnr': [], 'ssim': [], 'iou': []}}
    records = []

    for scat, mask, stem in test_dl:
        scat, mask = scat.to(device), mask.to(device)
        mask_np = mask.cpu().squeeze().numpy()
        scat_np = scat.cpu().squeeze().numpy()

        m_psnr = peak_signal_noise_ratio(mask_np, scat_np, data_range=1.0)
        m_ssim = structural_similarity(mask_np, scat_np, data_range=1.0)
        m_iou  = compute_iou(scat.cpu().squeeze(), mask.cpu().squeeze())
        metrics['scatter']['psnr'].append(m_psnr)
        metrics['scatter']['ssim'].append(m_ssim)
        metrics['scatter']['iou'].append(m_iou)

        with torch.no_grad():
            p_low = unet(scat)
        low_np = p_low.cpu().squeeze().numpy()
        u_psnr = peak_signal_noise_ratio(mask_np, low_np, data_range=1.0)
        u_ssim = structural_similarity(mask_np, low_np, data_range=1.0)
        u_iou  = compute_iou(p_low.cpu().squeeze(), mask.cpu().squeeze())
        metrics['unet']['psnr'].append(u_psnr)
        metrics['unet']['ssim'].append(u_ssim)
        metrics['unet']['iou'].append(u_iou)

        with torch.no_grad():
            res = dip_mod(p_low)
            hr  = torch.clamp(p_low + res, 0, 1)
        hr_np = hr.cpu().squeeze().numpy()
        d_psnr = peak_signal_noise_ratio(mask_np, hr_np, data_range=1.0)
        d_ssim = structural_similarity(mask_np, hr_np, data_range=1.0)
        d_iou  = compute_iou(hr.cpu().squeeze(), mask.cpu().squeeze())
        metrics['dtg']['psnr'].append(d_psnr)
        metrics['dtg']['ssim'].append(d_ssim)
        metrics['dtg']['iou'].append(d_iou)


        orig_img = to_pil(scat); gt_img = to_pil(mask)
        unet_img = to_pil(p_low); dtg_img = to_pil(hr)
        W, H = orig_img.size
        comp = Image.new('L', (W*4, H))
        for i, im in enumerate([orig_img, gt_img, unet_img, dtg_img]):
            comp.paste(im, (i*W, 0))
        comp.save(os.path.join(TEST_RESULTS_DIR, f"cmp_{stem}.png"))


        for method, psnr, ssim, iou in [
            ('scatter', m_psnr, m_ssim, m_iou),
            ('unet',    u_psnr, u_ssim, u_iou),
            ('dtg',     d_psnr, d_ssim, d_iou)
        ]:
            records.append({
                'sample': stem,
                'method': method,
                'psnr':   psnr,
                'ssim':   ssim,
                'iou':    iou
            })


    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(TEST_RESULTS_DIR, 'metrics.csv'), index=False)


    for m, vals in metrics.items():
        print(f"{m.upper()} Mean PSNR: {np.mean(vals['psnr']):.3f}, "
              f"SSIM: {np.mean(vals['ssim']):.3f}, "
              f"IoU: {np.mean(vals['iou']):.3f}")

if __name__ == '__main__':
    main()
