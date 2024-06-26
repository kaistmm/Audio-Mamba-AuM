# -*- coding: utf-8 -*-
# Modified version of the following file:
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, model_name='deit_base_distilled_patch16_384', verbose=True, _num_patches=None, 
                 ast_pretrain=False, ast_pretrain_path=None, ast_label_dim=527, ast_fstride=16, ast_tstride=16, ast_input_fdim=128, ast_input_tdim=1024, ast_model_name=None, load_backbone_only=False):

        super(ASTModel, self).__init__()
        # assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AST pretraining: {:s}'.format(str(imagenet_pretrain),str(ast_pretrain)))

        if 'distilled' in model_name:
            self.distilled = 1
        else:
            self.distilled = 0

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if ast_pretrain == False:
            self.v = timm.create_model(model_name, pretrained=imagenet_pretrain, embed_layer=PatchEmbed)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            
            # for accelerate multi-gpu training, delete the unused parameters
            del self.v.head

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 1 + self.distilled:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1 + self.distilled, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, (self.v.patch_embed.num_patches if _num_patches is None else _num_patches) + 1 + self.distilled, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif ast_pretrain == True:

            if ast_model_name is None:
                ast_model_name = model_name

            ast_model = ASTModel(
                label_dim=ast_label_dim,
                fstride=ast_fstride,
                tstride=ast_tstride,
                input_fdim=ast_input_fdim, 
                input_tdim=ast_input_tdim,
                model_name=ast_model_name,
                imagenet_pretrain=False,
                verbose=False
            )
            if ast_pretrain_path is not None:
                state_dict = torch.load(ast_pretrain_path, map_location='cpu')
                out_dict = {}
                for k, v in state_dict.items(): # Adjust the name of dict
                    if 'module.' in k:
                        out_dict[k[7:]] = v
                    elif not 'v.head' in k: # skip loading the unused parameters
                        out_dict[k] = v
                ast_model.load_state_dict(out_dict)
            
            self.v = ast_model.v
            ast_embed_dim = self.v.pos_embed.shape[2]

            self.original_embedding_dim = ast_embed_dim

            if load_backbone_only:
                self.mlp_head = nn.Sequential(nn.LayerNorm(ast_embed_dim), nn.Linear(ast_embed_dim, label_dim))
            else:
                if label_dim != ast_model.mlp_head[1].out_features:
                    raise ValueError('The label_dim of the ASTModel should be the same as the pretrained model.')
                self.mlp_head = ast_model.mlp_head
            
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            new_proj.weight = torch.nn.Parameter(ast_model.v.patch_embed.proj.weight)
            new_proj.bias = ast_model.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            ast_f_dim, ast_t_dim = self.get_shape(ast_fstride, ast_tstride, ast_input_fdim, ast_input_tdim)
            
            new_pos_embed = self.v.pos_embed[:, 1 + self.distilled:, :].detach().reshape(1, ast_f_dim * ast_t_dim, self.original_embedding_dim).transpose(1, 2).reshape(
                1, self.original_embedding_dim, ast_f_dim, ast_t_dim
            )
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= ast_t_dim:
                new_pos_embed = new_pos_embed[:, :, :, int(ast_t_dim / 2) - int(t_dim / 2): int(ast_t_dim / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(ast_f_dim, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= ast_f_dim:
                new_pos_embed = new_pos_embed[:, :, int(ast_f_dim / 2) - int(f_dim / 2): int(ast_f_dim / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1 + self.distilled, :].detach(), new_pos_embed], dim=1))
            
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # @autocast() # disabled because accelerate training configs already incorporate autocast
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B, _, F, T = x.shape
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        if T * F // (16 * 16) != self.v.pos_embed.shape[1]:
            pos_embed = self.v.pos_embed[:, 1:, :]
            pos_embed = pos_embed.transpose(1, 2).reshape(1, self.original_embedding_dim, F // 16, -1)
            target_size = (F // 16, T // 16)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=target_size, mode='bilinear')
            pos_embed = pos_embed.reshape(1, self.original_embedding_dim, -1).transpose(1, 2)
            pos_embed = torch.cat([self.v.pos_embed[:, :1, :], pos_embed], dim=1)
        else:
            pos_embed = self.v.pos_embed
        x = x + pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        if self.distilled:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]

        x = self.mlp_head(x)
        return x
