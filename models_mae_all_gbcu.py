# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import cv2

import torch
from torch import sigmoid
import torch.nn as nn
import einops

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from piqa import SSIM
from piqa.utils.functional import gaussian_kernel

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

criterion = SSIMLoss(reduction='none').cuda() # .cuda() if you need GPU support


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, overall_min, overall_max, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        self.bin_num = 256
        self.i = 0
        self.min_val = overall_min.detach().cpu().numpy().item()
        self.max_val = overall_max.detach().cpu().numpy().item()
        self.interval_length = (self.max_val - self.min_val) / self.bin_num
        self.L = 30
        
        self.device = 'cuda:0'
        self.ssim = criterion

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.get_decoder_depth = decoder_depth
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def custom_unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) # (64, 196, 768) -> (64, 196, 3 , 16 , 16)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        # print("x0->", x.shape)
        
        # x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        #For base
        x = x.reshape(shape=(x.shape[0], -1, p, p, 3))
        # For large
        # x = x.reshape(shape=(x.shape[0], -1, p, p, 4))


        # print("x1->", x.shape)
        # x = torch.einsum('nhwpqc->nchpwq', x)
        x = torch.einsum('nlpqc->nlcpq', x)

        # print("x2->", x.shape)
        # imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        # print("imgs->", imgs.shape)
        return x


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        print("decoder depth -> ",self.get_decoder_depth)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        
        # experimental Idea(not working)
        # x = torch.tanh(x) 
        # x = torch.clamp(x, min=self.min_val, max=self.max_val)

        return x
    
    def activation_func(self, img_flat, bins_av):
        # torch.set_default_device(self.device)

        img_minus_bins_av = torch.sub(img_flat, bins_av)  # shape=  (batch_size,H*W,bin_num)
        img_plus_bins_av = torch.add(img_flat, bins_av)  # shape = (batch_size,H*W,bin_num)
        # print("img_minus_bins_av", img_minus_bins_av.shape)
        self.kernel_width = self.interval_length / self.L
        
        maps = sigmoid((img_minus_bins_av + self.interval_length / 2) / self.kernel_width) \
               - sigmoid((img_minus_bins_av - self.interval_length / 2) / self.kernel_width) \
               + sigmoid((img_plus_bins_av - 2 * self.min_val + self.interval_length / 2) / self.kernel_width) \
               - sigmoid((img_plus_bins_av - 2 * self.min_val - self.interval_length / 2) / self.kernel_width) \
               + sigmoid((img_plus_bins_av - 2 * self.max_val + self.interval_length / 2) / self.kernel_width) \
               - sigmoid((img_plus_bins_av - 2 * self.max_val - self.interval_length / 2) / self.kernel_width)
        # print("maps->" , maps.shape)
        return maps
    
    
    def calc_activation_maps(self, img):
        # apply approximated shifted rect (bin_num) functions on img
        # torch.set_default_device(self.device)
        
        bins_min_max = np.linspace(self.min_val, self.max_val, self.bin_num + 1)
        bins_av = (bins_min_max[0:-1] + bins_min_max[1:]) / 2
        bins_av = torch.tensor(bins_av, device=img.device, dtype = torch.float32)  # shape = (,bin_num)
        bins_av = torch.unsqueeze(bins_av, 0)  # shape = (1,bin_num)
        bins_av = torch.unsqueeze(bins_av, 0)  # shape = (1,1,bin_num)
        bins_av = torch.unsqueeze(bins_av, 0)  # shape = (1,1,1,bin_num)

        # m = nn.Flatten()
        m = nn.Flatten(2, -1)

        img_flat = torch.unsqueeze(m(img), -1)
        # print("img_flat->", img_flat.shape)
        maps = self.activation_func(img_flat, bins_av)  # shape = (batch_size,H*W,bin_num)
        return maps

    def calc_glcm_maps(self, img):
        # hist_row = self.calc_activation_maps(img[:,:,:-1])
        hist_row = self.calc_activation_maps(img[:,:,:,:-1])
        # hist_shift_row = self.calc_activation_maps(img[:,:,1:])
        hist_shift_row = self.calc_activation_maps(img[:,:,:,1:])

        # glcm_row = torch.bmm(hist_row.permute(0, 2, 1), hist_shift_row)
        # glcm_row = torch.bmm(hist_row.permute(0, 1, 3, 2), hist_shift_row)
        glcm_row = torch.matmul(hist_row.permute(0, 1, 3, 2), hist_shift_row)
        
        #verify the multiplciation
        
        # hist_col = self.calc_activation_maps(img[:,:-1,:])
        hist_col = self.calc_activation_maps(img[:,:,:-1,:])

        # hist_shift_col = self.calc_activation_maps(img[:,1:,:])
        hist_shift_col = self.calc_activation_maps(img[:,:,1:,:])

        # glcm_col = torch.bmm(hist_col.permute(0, 2, 1), hist_shift_col)
        # glcm_col = torch.bmm(hist_col.permute(0, 1, 3, 2), hist_shift_col)
        glcm_col = torch.matmul(hist_col.permute(0, 1, 3, 2), hist_shift_col)

        #check if needed or not
        x = torch.einsum('nblxy-> blnxy', torch.stack([glcm_row, glcm_col]))
        # print("final glcm shape-> ", x.shape)
        
        return x 

        
    # def xlogy(self, x, y):
    #     z = torch.zeros(())
    #     # if torch.device.type == "cuda":
    #     z = z.to(self.device)
            
    #     print(x.get_device(), y.get_device(), z.get_device())

    #     return x * torch.where(x == 0., z, torch.log(y))

    def calc_cond_entropy_loss(self, maps_x, maps_y):
        # torch.set_default_device(self.device)

        pxy = torch.matmul(maps_x.permute(0,1,3,2), maps_y) / maps_x.shape[2]
        pxy = pxy.to(maps_x.device)
        py = torch.sum(pxy, 2)
        # py = py.to(self.device)
        # calc conditional entropy: H(X|Y)=-sum_(x,y) p(x,y)log(p(x,y)/p(y))
        hy = torch.sum(torch.xlogy(py, py), 2)
        hxy = torch.sum(torch.xlogy(pxy, pxy), [2, 3])
        cond_entropy = hy - hxy
        # mean_cond_entropy = torch.mean(cond_entropy)
        # return mean_cond_entropy
        return cond_entropy
    
    def calc_out_tar_map(self, out, tar):
        maps_out = self.calc_activation_maps(out)
        maps_tar = self.calc_activation_maps(tar)
        
        return maps_out, maps_tar
    
    def calc_glcm_out_tar(self, tar, out):
        glcm_tar = self.calc_glcm_maps(tar)
        glcm_out = self.calc_glcm_maps(out)
        
        return glcm_tar, glcm_out

    def forward_loss(self, imgs, pred, mask, is_scale_enabled=True):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
    
        """
        
        self.i += 1
        
            
        target = self.patchify(imgs)
        # pred_unpatch = self.unpatchify(pred)
        
        print(f"pred before min -> {pred.min()}, pred before max -> {pred.max()}")
        
        loss_rgb = (pred - target) ** 2
        loss_rgb = loss_rgb.mean(dim=-1)  # [N, L], mean loss_rgb per patch

        loss_rgb = (loss_rgb * mask).sum() / mask.sum()  # mean loss on removed patches
        
        # print("rgb_loss shape-> ", loss_rgb.shape)

        
        pred = torch.clamp(pred, min=self.min_val, max=self.max_val)
        
        pred_unpatch = self.custom_unpatchify(pred)
        print("target before shape", target.shape)
        target = self.custom_unpatchify(target)
        print("target shape->", target.shape)
        
        img_norm = target
        preds_norm = pred_unpatch
        
        batch_size, patches, _, height, width = target.shape
        
        img_norm = img_norm.reshape(img_norm.size(0), -1)
        img_norm -= img_norm.min(1, keepdim=True)[0]
        img_norm /= img_norm.max(1, keepdim=True)[0]
        img_norm = img_norm.reshape(batch_size, patches, -1, height, width)
        
        preds_norm = preds_norm.reshape(preds_norm.size(0), -1)
        preds_norm -= preds_norm.min(1, keepdim=True)[0]
        preds_norm /= preds_norm.max(1, keepdim=True)[0]
        preds_norm = preds_norm.reshape(batch_size, patches, -1, height, width)
        
        # img_norm -= img_norm.min(1, keepdim=True)[0]
        # img_norm /= img_norm.max(1, keepdim=True)[0]
        
        # preds_norm -= preds_norm.min(1, keepdim=True)[0]
        # preds_norm /= preds_norm.max(1, keepdim=True)[0]
        
        
        

        
        # ssim_loss = self.ssim(img_norm, preds_norm)
        # ssim_loss = ssim_loss.mean(dim=-1)  # [N, L], mean loss_glcm per patch

        # ssim_loss = (ssim_loss * mask).sum() / mask.sum()  # mean loss_glcm on removed patches

        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5
            
        img_norm = einops.rearrange(img_norm, 'b p c h w -> p b c h w ')
        preds_norm = einops.rearrange(preds_norm, 'b p c h w -> p b c h w ')

        print("img_norm shape->", img_norm.shape)

        ssim_loss = self.ssim(img_norm[0], preds_norm[0])
        ssim_loss = torch.unsqueeze(ssim_loss, dim=-1)
        
        # ----------------------------------------------------------------------------------------------------->
        for i in range(1, img_norm.shape[0]):
            ssim_loss_new = self.ssim(img_norm[i], preds_norm[i])
            ssim_loss_new = torch.unsqueeze(ssim_loss_new, dim=-1)

            ssim_loss = torch.concat([ssim_loss, ssim_loss_new], dim=-1)
        # ssim_loss = ssim_loss.mean(dim=-1)  # [N, L], mean loss_glcm per patch

        ssim_loss = (ssim_loss * mask).sum() / mask.sum()  
        # ----------------------------------------------------------------------------------------------------->
        
        # print("ssim_loss shape-> ", ssim_loss.shape)
        
            
        # print("Mask -> ", mask.shape)
        # print("Pred -> ", pred.shape)
        # print("Pred Unpatch ->", pred_unpatch.shape)
        # print("Target Unpatch-> ", target.shape)
        
        # plot pred_unpatch and target
        
        # pred_unpatch
        
        glcm_1 = self.calc_glcm_out_tar(pred_unpatch[:,:,0,:,:], target[:,:,0,:,:])
        # glcm_2 = self.calc_glcm_out_tar(pred_unpatch[:,:,1,:,:], target[:,:,1,:,:])
        # glcm_3 = self.calc_glcm_out_tar(pred_unpatch[:,:,2,:,:], target[:,:,2,:,:])


        # loss_1 = self.calc_cond_entropy_loss(glcm_1[0][:,:,0,:,:], glcm_1[1][:,:,0,:,:])
        # loss_2 = self.calc_cond_entropy_loss(glcm_1[0][:,:,1,:,:], glcm_1[1][:,:,1,:,:])

        #----------------->>
        if is_scale_enabled:
            glcm_1_sum = torch.sum(glcm_1[0], (-2, -1))
            glcm_1_sum = glcm_1_sum.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,self.bin_num,self.bin_num)
            glcm_new_1 = glcm_1[0] * torch.div((glcm_1_sum - glcm_1[0]), glcm_1_sum)
            
            glcm_1_sum = torch.sum(glcm_1[1], (-2, -1))
            glcm_1_sum = glcm_1_sum.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,self.bin_num,self.bin_num)
            glcm_new_2 = glcm_1[1] * torch.div((glcm_1_sum - glcm_1[1]), glcm_1_sum)
            
            loss = (glcm_new_1 - glcm_new_2) ** 2

        else:
            loss = (glcm_1[0] - glcm_1[1]) ** 2
        # loss = (glcm_1[0] - glcm_1[1]) 
        #----------------->>


        loss = einops.rearrange(loss, 'b l n x y -> b l (n x y)')
        # print("final loss shape->", loss.shape)


        # loss = (pred - target) ** 2
        
        #------------------>>
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        #------------------>>

        # print("glcm_loss shape-> ", lossy.shape)

        
        # loss_1 = (loss_1 * mask).sum() / mask.sum()  # mean loss on removed patches
        # loss_2 = (loss_2 * mask).sum() / mask.sum()  # mean loss on removed patches

        # loss = (loss_1 + loss_2) / 2
        
        y = self.unpatchify(pred)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
        
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        x = torch.einsum('nchw->nhwc', imgs)
        x = x.to('cpu')

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask
        
        print(f"pred min-> {pred_unpatch.min()}, pred max-> {pred_unpatch.max()}")
        print(f"img min-> {target.min()}, img max-> {target.max()}")


        to_write = x[0].detach().cpu().numpy()
        to_write_rec = (y[0].detach().cpu().numpy())
        to_write_rec_original = im_paste[0].numpy()
        
        to_write = (((to_write - to_write.min()) / (to_write.max() - to_write.min())) * 255.9)
        to_write_rec = (((to_write_rec - to_write_rec.min()) / (to_write_rec.max() - to_write_rec.min())) * 255.9)
        to_write_rec_original = (((to_write_rec_original - to_write_rec_original.min()) / (to_write_rec_original.max() - to_write_rec_original.min())) * 255.9)
        
        # if self.i % 100 == 0:

            # cv2.imwrite(f"generated/models_mae_hist_ssim_rgb_gbcu/{self.i}.png", to_write)
            # cv2.imwrite(f"generated/models_mae_hist_ssim_rgb_gbcu/{self.i}_rec.png", to_write_rec)
            # cv2.imwrite(f"generated/models_mae_hist_ssim_rgb_gbcu/{self.i}_rec_original.png", to_write_rec_original)

        return loss + ssim_loss + 0.1*loss_rgb
        # return loss + 0.1*loss_rgb


    def forward(self, imgs, epoch, max_epochs, mask_ratio=0.75):
        # self.L = self.get_current_L(epoch, max_epochs)
        self.L = 30
        print("wefewf", self.L, epoch)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def get_current_L(self, epoch, max_epochs):
        base_L = 5
        curr_L = base_L + (epoch-200) // 6
        return curr_L

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_decoder10(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=10, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16_d10 = mae_vit_base_patch16_decoder10
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks