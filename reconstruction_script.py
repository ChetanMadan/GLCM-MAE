import cv2
import torch
import os
from PIL import Image
import numpy as np
import models_mae
import argparse

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_dir', type=str, default=None, help='Path to the chkpt file')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data dir')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to the save directory')
    args = parser.parse_args()
    return args


def prepare_model(chkpt_dir, data_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def get_reconstructions(chkpt_dir, data_dir, save_dir):
    img_path = get_image_paths_from_dir(data_dir)
    # img_path = ["/home/aarjav/scratch/busi/splits/imgs/benign_22.png", "/home/aarjav/scratch/busi/splits/imgs/benign_23.png", "/home/aarjav/scratch/busi/splits/imgs/benign_24.png", "/home/aarjav/scratch/busi/splits/imgs/benign_38.png", "/home/aarjav/scratch/busi/splits/imgs/malignant_34.png", "/home/aarjav/scratch/busi/splits/imgs/malignant_61.png"]
    # img_path = ["/home/aarjav/scratch/gbcu_data/imgs/images/im00064.jpg", "/home/aarjav/scratch/gbcu_data/imgs/images/im00027.jpg", "/home/aarjav/scratch/gbcu_data/imgs/images/im00512.jpg", "/home/aarjav/scratch/gbcu_data/imgs/images/im01117.jpg"]
    model = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    
    os.makedirs(save_dir, exist_ok=True)
    
    for img in img_path:
        img_p = img
        img = Image.open(img)
        img = img.resize((224, 224))
        img = np.array(img) / 255.

        img = np.stack([img, img, img], axis=-1)
        assert img.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        x = torch.tensor(img)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # run MAE
        loss, y, mask = model(x.float(), mask_ratio=0.75)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
        
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # print(y_1.shape)
        
        x = torch.clip((x * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        x = x.squeeze(0)
        
        y = torch.clip((y * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        y = y.squeeze(0)
        
        im_masked = torch.clip((im_masked * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        im_masked = im_masked.squeeze(0)
        
        im_paste = torch.clip((im_paste * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        im_paste = im_paste.squeeze(0)


        y = y.detach().cpu().numpy()
        im_masked = im_masked.detach().cpu().numpy()
        im_paste = im_paste.detach().cpu().numpy()
        x = x.detach().cpu().numpy()


        # print(y_1.shape)
        cv2.imwrite(os.path.join(save_dir, img_p.split('/')[-1].split('png')[0]) + '_original.png', x)
        cv2.imwrite(os.path.join(save_dir, img_p.split('/')[-1].split('png')[0]) + '_recon.png', y)
        cv2.imwrite(os.path.join(save_dir, img_p.split('/')[-1].split('png')[0]) + '_recon_ori.png', im_paste)
        cv2.imwrite(os.path.join(save_dir, img_p.split('/')[-1].split('png')[0]) + '_mask.png', im_masked)
        
        print('next')


def main():
    args = parse_args_and_config()
    chkpt_dir = args.chkpt_dir
    data_dir = args.data_dir
    save_dir = args.save_dir

    get_reconstructions(chkpt_dir, data_dir, save_dir)
 


if __name__ == '__main__':
    main()