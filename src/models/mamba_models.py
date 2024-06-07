import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

from src.utilities.rope import * 
from src.utilities.tokenization import FlexiPatchEmbed, FlexiPosEmbed, PatchEmbed, resample_patch_embed, vanilla_resample_patch_embed
from torch.cuda.amp import autocast
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class AudioMamba(nn.Module):
    def __init__(self, 
                 spectrogram_size=(128, 1024),
                 patch_size=(16, 16),
                 strides=(16, 16),
                 depth=24, 
                 embed_dim=768,
                 channels=1,
                 num_classes=527,
                 ssm_cfg=None, 
                 drop_rate=0., # A
                 drop_path_rate=0, # A
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm=True, 
                 residual_in_fp32=True, 
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 abs_pos_patch_grid_size=None,
                 pt_hw_seq_len=None,
                 final_pool_type='mean',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 if_cls_token=True,
                 imagenet_pretrain=False,
                 imagenet_pretrain_path=None,
                 imagenet_pretrain_modelkey='model',
                 aum_pretrain=False,
                 aum_pretrain_path=None,
                 aum_pretrain_fstride=None,
                 aum_pretrain_tstride=None,
                 bilinear_rope=False,
                 flip_img_sequences_ratio=-1.,
                 if_bidirectional=False,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_devide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 imagenet_load_double_cls_token=False,
                 imagenet_load_middle_cls_token=True,
                 weird__target_size=None,
                 must_square=False,
                 transpose_token_sequence=False,
                 use_end_cls_token=False,
                 use_PI_for_patch_embed=True,
                 flexible_patch_sizes=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.use_end_cls_token = use_end_cls_token
        self.num_tokens = 0
        self.spectrogram_size = spectrogram_size
        self.patch_size = to_2tuple(patch_size)
        self.strides = strides
        self.channels = channels
        self.embed_dim = embed_dim
        self.pt_hw_seq_len = pt_hw_seq_len
        self.ft_seq_len = ft_seq_len
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.weird__target_size = weird__target_size
        self.must_square = must_square
        self.transpose_token_sequence = transpose_token_sequence

        if must_square:
            assert spectrogram_size[1] % spectrogram_size[0] == 0, "The spectrogram size must be a square"
            dv = spectrogram_size[1] // spectrogram_size[0]
            side = int(dv ** 0.5)
            assert side ** 2 == dv, "The spectrogram size must be a square"
            self.patch_grid_size = FlexiPosEmbed.get_shape(*strides, patch_size, spectrogram_size[0] * side, spectrogram_size[1] // side)
        else:
            self.patch_grid_size = FlexiPosEmbed.get_shape(*strides, patch_size, *spectrogram_size)
        
        self.num_patches = self.patch_grid_size[0] * self.patch_grid_size[1]
        

        # TODO: Add a checker that looks at the patch size and spectrogram size to make sure they are compatible

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 1

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.head.apply(segm_init_weights)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        if if_rope and bilinear_rope:
            self.init_rope()

        proj_load = None
        pos_embed_load = None
        pos_grid_size_load = (-1, -1)

        if imagenet_pretrain:
            weights = torch.load(imagenet_pretrain_path, map_location='cpu')[imagenet_pretrain_modelkey]
        
            weights['pos_embed.pos_embed'] = weights['pos_embed']
            del weights['pos_embed']

            if self.channels == 1:
                weights['patch_embed.proj.weight'] = weights['patch_embed.proj.weight'].mean(1, keepdim=True)
            
            proj_load = Namespace(
                weight=weights['patch_embed.proj.weight'],
                bias=weights['patch_embed.proj.bias'],
            )

            pos_embed_load = weights['pos_embed.pos_embed']

            # NOTE: This assumes that the imagenetpretrained model never relocates the positional embeddings of the cls tokens to the beginning
            # keeping them naturally at their corresponding positions
            # NOTE: This assumes that we are always loading from the same cls token setting as the current model
            if imagenet_load_double_cls_token:
                pos_embed_load = FlexiPosEmbed.insert_to_prefix(pos_embed_load, [0, pos_embed_load.shape[1] - 1])
            elif imagenet_load_middle_cls_token:
                # bring the middle pos embed into the first position
                N = pos_embed_load.shape[1] - 1 # This N represents the N in the forward pass
                pos_embed_load = FlexiPosEmbed.insert_to_prefix(pos_embed_load, N//2)
            
            pos_grid_size_load = to_2tuple(int((weights['pos_embed.pos_embed'].shape[1] - self.num_tokens) ** 0.5))

            if self.pt_hw_seq_len is None: # If this is None, then the vim model is trained from scratch
                self.pt_hw_seq_len = to_2tuple(pos_grid_size_load)

            if if_rope:
                if bilinear_rope:
                    weights['rope.freqs_cos'] = self.interp_rope(weights['rope.freqs_cos'], pos_grid_size_load)
                    weights['rope.freqs_sin'] = self.interp_rope(weights['rope.freqs_sin'], pos_grid_size_load)
                else:
                    del weights['rope.freqs_cos']
                    del weights['rope.freqs_sin']

            del weights['pos_embed.pos_embed']
            del weights['patch_embed.proj.weight']
            del weights['patch_embed.proj.bias']
            del weights['head.weight']
            del weights['head.bias']

            print(self.load_state_dict(weights, strict=False))

        if aum_pretrain:
            weights = torch.load(aum_pretrain_path, map_location='cpu')
            # remove data parallel prefix
            weights = {k.replace("module.", ""): v for k, v in weights.items()}

            proj_load = Namespace(
                weight=weights['patch_embed.proj.weight'],
                bias=weights['patch_embed.proj.bias'],
            )

            patch_size_load = proj_load.weight.shape[-2:]
            
            if aum_pretrain_fstride is None:
                aum_pretrain_fstride = patch_size_load[0]
            
            if aum_pretrain_tstride is None:
                aum_pretrain_tstride = patch_size_load[1]

            strides_load = (aum_pretrain_fstride, aum_pretrain_tstride)

            pos_embed_load = weights['pos_embed.pos_embed']

            if must_square:
                assert False, "Double check here"
            for log_audio_length in range(6, 20):
                pos_grid_size_load = FlexiPosEmbed.get_shape(*strides_load, patch_size_load, 128, 2**log_audio_length)
                if pos_grid_size_load[0] * pos_grid_size_load[1] == pos_embed_load.shape[1] - self.num_tokens:
                    break
                if log_audio_length == 19:
                    raise ValueError("Could not find matching audio length")
            
            if self.pt_hw_seq_len is None: # If the pt_hw_seq_len is None, the aum model we load from is trained from scratch
                self.pt_hw_seq_len = to_2tuple(pos_grid_size_load)
            
            if if_rope:
                if bilinear_rope:
                    weights['rope.freqs_cos'] = self.interp_rope(weights['rope.freqs_cos'], pos_grid_size_load)
                    weights['rope.freqs_sin'] = self.interp_rope(weights['rope.freqs_sin'], pos_grid_size_load)
                else:
                    del weights['rope.freqs_cos']
                    del weights['rope.freqs_sin']

            del weights['pos_embed.pos_embed'] 
            del weights['patch_embed.proj.weight']
            del weights['patch_embed.proj.bias']

            # check if num_classes is the same
            if weights['head.weight'].shape[0] != num_classes:
                print('Num classes differ! Can only load the backbone weights.')
                del weights['head.weight']
                del weights['head.bias']

            print(self.load_state_dict(weights, strict=False))

        if if_rope and not bilinear_rope: 
            self.init_rope()

        self.patch_embed = FlexiPatchEmbed( # For now, let's use a simple patchembed module
            patch_size=patch_size,
            strides=strides,
            in_chans=channels,
            embed_dim=embed_dim,
            proj_load=proj_load,
            resize_func=resample_patch_embed if use_PI_for_patch_embed else vanilla_resample_patch_embed,
            precompute_for=flexible_patch_sizes,
        )

        if if_abs_pos_embed:
            self.pos_embed = FlexiPosEmbed(
                input_size=spectrogram_size,
                patch_size=patch_size,
                strides=strides,
                pos_grid_size=self.patch_grid_size if abs_pos_patch_grid_size is None else abs_pos_patch_grid_size,
                embed_dim=embed_dim,
                n_prefix_tokens=self.num_tokens,
                pos_embed_load=pos_embed_load,
                pos_grid_size_load=pos_grid_size_load,
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

    def interp_rope(self, weights, load_grid_size):
        weights = weights.view(1, load_grid_size[0], load_grid_size[1], -1).permute(0, 3, 1, 2)
        target_grid_size = self.patch_grid_size
        weights = nn.functional.interpolate(weights, size=target_grid_size, mode='bilinear')
        weights = weights.permute(0, 2, 3, 1).view(target_grid_size[0] * target_grid_size[1], -1)
        return weights

    def init_rope(self):
        half_head_dim = self.embed_dim // 2
            
        if self.pt_hw_seq_len is None: # If pt_hw_seq_len is None, then the model is being trained from scratch
            self.pt_hw_seq_len = self.patch_grid_size
        
        self.rope = VisionRotaryEmbedding(
            dim=half_head_dim,
            pt_seq_len=self.pt_hw_seq_len,
            ft_seq_len=self.patch_grid_size if self.ft_seq_len is None else self.ft_seq_len,
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False, patch_size=None, strides=None):
        x = x.unsqueeze(1) # B x C=1 x T x F
        x = x.transpose(2, 3) # B x C=1 x F x T

        B, C, F, T = x.shape
        
        if self.must_square:
            dv = T // F
            x = x.reshape(B, C, F, dv, F)
            x = x.transpose(2, 3) # B x C x dv x F x F
            side = int(dv ** 0.5)
            x = x.reshape(B, C, side, side, F, F)
            x = x.permute(0, 1, 2, 4, 3, 5).reshape(B, C, side * F, side * F)

        x = self.patch_embed(x, patch_size=patch_size, strides=strides)
        B, N, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, N + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
            else:
                cls_token = self.cls_token.expand(B, -1, -1)
                if if_random_cls_token_position:
                    token_position = random.randint(0, N)
                elif self.use_middle_cls_token:
                    token_position = N // 2
                elif self.use_end_cls_token:
                    token_position = N
                else:
                    token_position = 0
                x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
            N = x.shape[1]
        else:
            token_position = None

        if self.if_abs_pos_embed:
            x = self.pos_embed(x, token_position=token_position, target_size=self.weird__target_size, patch_size=patch_size, strides=strides)
            x = self.pos_drop(x)

        if self.transpose_token_sequence:
            # import ipdb; ipdb.set_trace()
            if self.if_cls_token:
                if self.use_double_cls_token:
                    head_t, tail_t = x[:, 0, :].unsqueeze(1), x[:, -1, :].unsqueeze(1)
                    x = x[:, 1:-1, :]
                else:
                    t = x[:, token_position, :].unsqueeze(1)
                    x = torch.cat((x[:, :token_position, :], x[:, token_position + 1:, :]), dim=1)
            
            # reshape x to be B x F x T x D
            _F, _T = F // self.patch_size[0], T // self.patch_size[1]
            x = x.reshape(B, _F, _T, -1)
            x = x.transpose(1, 2)
            x = x.reshape(B, _T * _F, -1)

            if self.if_cls_token:
                if self.use_double_cls_token:
                    x = torch.cat((head_t, x, tail_t), dim=1)
                else:
                    x = torch.cat((x[:, :token_position, :], t, x[:, token_position:, :]), dim=1)

        if if_random_token_rank:

            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(N)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)



        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                return hidden_states[:, token_position, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    # @autocast()
    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False, patch_size=None, strides=None): # NOTE: For now, these are all being used as default. Later, these could be set through the args param
        x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank, patch_size=patch_size, strides=strides)
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x
