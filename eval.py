import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff

from unet import UNet
import logging
import numpy as np
import matplotlib.pyplot as plt



def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()
                if net.n_classes > 1:
                    tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return tot / n_val
    
def out_net(net, in_image, logging):
    checkpoint_path = '/home/qgao10/checkpoints/CP_epoch1.pth'
    net.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
    logging.info(f'Model loaded from {checkpoint_path}')
    
    net.eval()
    
    mask_pred = net(in_image)
    out_img = mask_pred.detach().numpy()[0][0]

    np.save('/home/qgao10/out_img', out_img)
    print('out saved')

    
    return out_img
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    img_path = '/home/qgao10/modis_data/0_Image_067.npy'
    img = np.load(img_path)
    img = np.array([[img[31],img[14],img[25]]])
    in_img = torch.from_numpy(img)
    
    net = UNet(n_channels=3, n_classes=1)
    out_img = out_net(net, in_img, logging)
    
    # plt.imshow(out_img)
    
