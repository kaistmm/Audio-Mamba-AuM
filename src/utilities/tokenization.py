# From FlexiAST: https://arxiv.org/pdf/2307.09286.pdf

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import numpy as np
import timm
from timm.models.layers import to_2tuple
import torch.nn.functional as F
from typing import List, Optional
import math
import copy
from torch.cuda.amp import autocast
from timm.models.layers import trunc_normal_, lecun_normal_

def divs(n):
    return [i for i in range(1, n + 1) if n % i == 0]

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bilinear',
        antialias: bool = True, # Google uses True (implicitly)
        verbose: bool = False,
        pos_embed_prefix=True,
):
    # sort out sizes, assume square if old size not provided
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens > 0 and pos_embed_prefix: # TODO: CHECK THIS!!!
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation

    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    if verbose:
        # TODO: Implement logging here
        # _logger.info(f'Resized position embedding: {old_size} to {new_size}.')
        pass

    # add back extra (class, etc) prefix tokens
    if num_prefix_tokens > 0 and pos_embed_prefix:
        if verbose:
            print(posemb_prefix.shape, posemb.shape)
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb

def get_resize_mat_pinv(
    old_size: List[int],
    new_size: List[int], 
    interpolation: str = 'bilinear',
    antialias: bool = False,
):
    
    import numpy as np
    assert len(old_size) == 2, "Old shape should only be hw"
    assert len(new_size) == 2, "New shape should only be hw"
    
    if tuple(old_size) == tuple(new_size):
        return torch.eye(np.prod(old_size))

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size) # This might be the B mentioned in the paper.

    try:
        resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))
    except:
        resize_mat_pinv = torch.linalg.pinv(torch.Tensor(resize_mat.T))

    return resize_mat_pinv

def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bilinear',
        antialias: bool = False,
        resize_mat_pinv=None,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """

    old_size = patch_embed.shape[-2:]

    if old_size == new_size:
        return patch_embed

    if resize_mat_pinv is None:
        resize_mat_pinv = get_resize_mat_pinv(
            old_size=old_size,
            new_size=new_size,
            interpolation=interpolation,
            antialias=antialias,
        ).detach()

    # new^2 old^w,768 1 old^2 -> 768 1 new^2
    ens = torch.einsum('xk,abk->abx', [
        resize_mat_pinv.to(patch_embed.device),
        patch_embed.reshape(patch_embed.size(0),patch_embed.size(1), -1)
    ]).reshape(patch_embed.size(0), patch_embed.size(1), new_size[0], new_size[1])
    return ens

def vanilla_resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bilinear',
        antialias: bool = True # Google uses True (implicitly)
    ):

    B, C, H, W = patch_embed.shape

    new_size = to_2tuple(new_size)
    old_size = to_2tuple((H, W))
    if new_size == old_size:  # might not both be same container type
        return patch_embed

    # do the interpolation
    patch_embed = F.interpolate(patch_embed, size=new_size, mode=interpolation, antialias=antialias)

    return patch_embed

def get_shape(fstride, tstride, patch_size, input_fdim=128, input_tdim=1024):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = nn.Conv2d(1, 768, kernel_size=(patch_size, patch_size), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return [f_dim, t_dim]

def resize_weights(weights,ori_patch,new_patch,methods,t_length,s_length):
    """This funtion is different with change_model, which is to transfer the original weights to target shape
    so that it can be loaded by another new model with different patch size. It means the model should keep the dimension the same.
    """
    # Fristly, deal with the pos_embed.
    pos_emb = weights["module.v.pos_embed"] # [1, 513, 768]
    ori_size = get_shape(ori_patch,ori_patch,ori_patch,input_tdim=t_length)
    new_size = get_shape(new_patch,new_patch,new_patch,input_tdim=s_length)
    # import ipdb;ipdb.set_trace()
    pos_emb = resample_abs_pos_embed(pos_emb,
                        new_size=new_size,
                        old_size=ori_size,
                        num_prefix_tokens=1,
                        verbose=True)
    weights["module.v.pos_embed"] = pos_emb


    p_weight = weights["module.v.patch_embed.proj.weight"]
    if methods == "PI":
        print("Use PI Resize")
        weights["module.v.patch_embed.proj.weight"] = resample_patch_embed(p_weight,(new_patch,new_patch))
    elif methods == "BL":
        print("Use Bilinear Resize")
        weights["module.v.patch_embed.proj.weight"] = vanilla_resample_patch_embed(p_weight,(new_patch,new_patch))

    return weights

def load_for_distill(ori_weights,student,teacher,s_patch,t_patch,methods,initial=True, model='ast'):
    out_dict = {}
    if methods != "SC" and model == 'ast':
        resized_weights = resize_weights(copy.deepcopy(ori_weights),t_patch,s_patch,methods) # should transform teacher's params for student to load
        for k, v in resized_weights.items(): # Adjust the name of dict
            out_dict[k[7:]] = v
        student.load_state_dict(out_dict)
    else:
        if model == 'flexiast':
            print('Distill flexi ast model')
        else:
            print("Distill student from scratch")

    out_dict = {}
    for k, v in ori_weights.items(): # Adjust the name of dict
        out_dict[k[7:]] = v
    teacher.load_state_dict(out_dict)
    return student,teacher

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=(16, 16),
        in_chans=1, 
        embed_dim=768,
    ):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=(16, 16),
        strides=(16, 16),
        in_chans=1, 
        embed_dim=768,
        bias=True, 
        norm_layer=None,
        flatten=True,
        proj_load=None,
        resize_func=resample_patch_embed,
        precompute_for=None,
        verbose=True
    ):
        super().__init__()

        print(f'Resize function is {resize_func.__name__}')

        if verbose:
            print(f'Initializing FlexiPatchEmbed with the following parameters:')
            print(f'patch_size={patch_size}, in_chans={in_chans}, embed_dim={embed_dim}, bias={bias}, norm_layer={norm_layer}, flatten={flatten}, proj_load={"yes" if proj_load is not None else None}, resize_func={resize_func.__name__}')
        
        self.patch_size = to_2tuple(patch_size)
        self.strides = to_2tuple(strides)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=strides, bias=bias)
        self.resize_func = resize_func

        lecun_normal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if verbose:
            print(f'The resize function is {self.resize_func.__name__}')

        if proj_load is not None:

            if verbose:
                print(f'Loading projection weights!')
                print(f'The shapes of the current projection: bias={self.proj.bias.shape}, weight={self.proj.weight.shape}')
                print(f'The shapes of the loaded projection: bias={proj_load.bias.shape}, weight={proj_load.weight.shape}')

            if proj_load.bias.shape != self.proj.bias.shape:
                raise ValueError("The bias shape of the loaded projection layer does not match the current projection layer")
            
            # copy the bias
            self.proj.bias = nn.Parameter(proj_load.bias)

            if proj_load.weight.shape != self.proj.weight.shape:
                self.proj.weight = nn.Parameter(self.resize_func(
                    proj_load.weight,
                    list(self.patch_size)
                ))
                if verbose:
                    print(f'Resized the projection weights with {self.resize_func.__name__}')
                    print(f'The shapes of the resized projection weights={self.proj.weight.shape}')
            else:
                self.proj.weight = nn.Parameter(proj_load.weight)

        self.precomputed_matrices = {}

        if precompute_for is not None:
            if type(precompute_for) is not list:
                raise ValueError("The precompute_for should be either None or a list!")             
            
            if self.resize_func.__name__ != resample_patch_embed.__name__:
                raise ValueError("The precompute_for is only supported when the resize_func is resample_patch_embed!")

            precompute_for = [to_2tuple(patch_size) for patch_size in list(precompute_for)]
            
            for patch_size in precompute_for:
                self.precomputed_matrices[patch_size] = get_resize_mat_pinv(
                    list(self.patch_size),
                    list(patch_size),
                ).detach()
            
            if verbose:
                print(f'Precomputed weights for {precompute_for}')

    def forward(self, x, patch_size=None, strides=None):
        B, C, H, W = x.shape
        
        if patch_size is None:
            patch_size = self.patch_size
        patch_size = to_2tuple(patch_size)

        if strides is None:
            strides = self.strides
        strides = to_2tuple(strides)

        if patch_size == self.patch_size:
            weight = self.proj.weight
        elif patch_size in self.precomputed_matrices:
            weight = self.resize_func(
                self.proj.weight,
                list(patch_size),
                resize_mat_pinv = self.precomputed_matrices[patch_size]
                    .to(x.device)
            )
        else:
            weight = self.resize_func(
                self.proj.weight,
                list(patch_size)
            )

        bias = self.proj.bias

        x = F.conv2d(x, weight, bias=bias, stride=strides)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class FlexiPosEmbed(nn.Module):
    def __init__(
        self,
        input_size=(128, 1024),
        patch_size=(16, 16),
        strides=(16, 16),
        pos_grid_size=(8, 64),
        embed_dim=768,
        pos_embed_load=None,
        pos_grid_size_load=None,
        n_prefix_tokens=1, # Assuming there is a cls token by default
        pos_embed_prefix=True,
        verbose=True
    ):
        super().__init__()

        if verbose:
            print(f'Initializing FlexiPosEmbed with the following parameters:')
            print(f'input_size={input_size}, pos_grid_size={pos_grid_size}, embed_dim={embed_dim}, pos_embed_load={pos_embed_load.shape if pos_embed_load is not None else None}, pos_grid_size_load={pos_grid_size_load}, n_prefix_tokens={n_prefix_tokens}, pos_embed_prefix={pos_embed_prefix}')

        self.input_size = to_2tuple(input_size)
        self.strides = to_2tuple(strides)
        self.patch_size = to_2tuple(patch_size)

        if pos_grid_size is None:
            self.pos_grid_size = to_2tuple(FlexiPosEmbed.get_shape(*strides, patch_size, *input_size))
        else:
            self.pos_grid_size = to_2tuple(pos_grid_size)

        pos_grid_size_load = to_2tuple(pos_grid_size_load)

        num_patches = self.pos_grid_size[0] * self.pos_grid_size[1]
        self.n_prefix_tokens = n_prefix_tokens
        self.pos_embed_prefix = pos_embed_prefix
        self.embed_dim = embed_dim
        pos_embed_shape = (1, num_patches + (n_prefix_tokens if pos_embed_prefix else 0), embed_dim)

        if pos_embed_load is not None:

            if verbose:
                print(f'Loading position embedding!')
                print(f'The shape of the current grid size: {pos_grid_size}')
                print(f'The shape of the loaded grid size: {pos_grid_size_load}')

            if pos_grid_size_load is None:
                raise ValueError("The loaded position embedding does not have the grid size information")
            
            if pos_grid_size != pos_grid_size_load:
                self.pos_embed = nn.Parameter(
                    resample_abs_pos_embed(
                        pos_embed_load,
                        new_size=self.pos_grid_size,
                        old_size=pos_grid_size_load,
                        num_prefix_tokens=n_prefix_tokens,
                        pos_embed_prefix=self.pos_embed_prefix
                    )
                )
                if verbose:
                    print(f'The shape of the resampled position embedding: {self.pos_embed.shape}')
            else:
                self.pos_embed = nn.Parameter(pos_embed_load)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(*pos_embed_shape))
            self.pos_embed = trunc_normal_(self.pos_embed, std=.02)
            
    
    @staticmethod
    def get_shape(fstride, tstride, patch_size, input_fdim, input_tdim):
        patch_size = to_2tuple(patch_size)
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, 1, kernel_size=patch_size, stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    @staticmethod
    def insert_to_prefix(x, from_poses):
        if type(from_poses) is int:
            from_poses = [from_poses]
        for i, from_pos in enumerate(from_poses):
            x = torch.cat([
                x[:, :i],
                x[:, from_pos:from_pos+1],
                x[:, i:from_pos],
                x[:, from_pos+1:]
            ], dim=1)
        return x

    @staticmethod
    def insert_from_prefix(x, to_poses):
        if type(to_poses) is int:
            to_poses = [to_poses]
        x_prefix, x = x[:, :len(to_poses)], x[:, len(to_poses):]
        for i, to_pos in enumerate(to_poses):
            x = torch.cat([
                x[:, :to_pos],
                x_prefix[:, i:i+1],
                x[:, to_pos:]
            ], dim=1)
        return x

    def forward(self, x, patch_size=None, strides=None, token_position=None, target_size=None):
        
        if token_position is not None:
            x = FlexiPosEmbed.insert_to_prefix(x, from_poses=token_position)

        if patch_size is None and strides is None and target_size is None:
            x = x + self.pos_embed
        elif target_size is not None:
            forward_pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=target_size,
                old_size=self.pos_grid_size, 
                num_prefix_tokens=self.n_prefix_tokens,
                pos_embed_prefix=self.pos_embed_prefix
            )
            if not self.pos_embed_prefix:
                final_patches = (
                    x[:, :self.n_prefix_tokens], (x[:, self.n_prefix_tokens:] + forward_pos_embed)
                )
                x = torch.cat(final_patches, dim=1)
            else:
                x = x + forward_pos_embed
        else:
            if patch_size is None:
                patch_size = self.patch_size
            patch_size = to_2tuple(patch_size)
            
            if strides is None:
                strides = self.strides
            strides = to_2tuple(strides)
            
            old_size = self.pos_grid_size
            new_size = [*FlexiPosEmbed.get_shape(*strides, patch_size, *self.input_size)]
            forward_pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=new_size,
                old_size=old_size, 
                num_prefix_tokens=self.n_prefix_tokens,
                pos_embed_prefix=self.pos_embed_prefix
            )

            if not self.pos_embed_prefix:
                final_patches = (
                    x[:, :self.n_prefix_tokens], (x[:, self.n_prefix_tokens:] + forward_pos_embed)
                )
                x = torch.cat(final_patches, dim=1)
            else:
                x = x + forward_pos_embed
        
        if token_position is not None:
            x = FlexiPosEmbed.insert_from_prefix(x, to_poses=token_position)
        
        return x
