import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate_acc
from accelerate import Accelerator
import random
from fvcore.common.config import CfgNode
from epic_sounds.epic_data import loader
import yaml
from utilities import *

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  

exp_seeds = [3949, 907, 1592, 99, 1453, 33, 1881, 2001, 102]
#              0    1     2    3   4    5    6     7     8
set_seed(exp_seeds[0])

accelerator = Accelerator()

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--exp-name", type=str, default="", help="experiment name")
parser.add_argument('-w', '--num-workers', default=4, type=int, metavar='NW', help='# of workers for dataloading')
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument("--run_type", type=str, default='train', help="run type", choices=["train", "eval"])

# Data
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json (only for speechcommands)")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')
parser.add_argument("--melbins", type=int, default=128, help="melbins")
parser.add_argument("--fshift", type=int, default=10, help="Frame shift (in ms)")

# Model General
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--model_type", type=str, default='base_dist_384', help="the model type used")
parser.add_argument("--fpatch_size", type=int, default=16, help="patch size in the frequency axis")
parser.add_argument("--tpatch_size", type=int, default=16, help="patch size in the time axis")
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained model', type=ast.literal_eval, default='False')
parser.add_argument('--imagenet_pretrain_path', help='the path to the ImageNet pretrained model', type=str, default=None)
parser.add_argument('--imagenet_pretrain_modelkey', help='the key of the ImageNet pretrained model (only for ViM)', type=str, default='model')

# AuM
parser.add_argument("--aum_pretrain", type=ast.literal_eval, default='False', help="If initialize with pretrained aum model")
parser.add_argument("--aum_pretrain_path", type=str, default=None, help="pretrain aum model path")
parser.add_argument("--aum_pretrain_fstride", type=int, default=16, help="pretrain aum model fstride (for positional embeddings)")
parser.add_argument("--aum_pretrain_tstride", type=int, default=16, help="pretrain aum model tstride (for positional embeddings)")
parser.add_argument("--pt_seq_lenf", type=int, default=None, help="pretrain sequence length in the frequency axis (used for the rope embeddings)")
parser.add_argument("--pt_seq_lent", type=int, default=None, help="pretrain sequence length in the time axis (used for the rope embeddings)")
parser.add_argument("--bilinear_rope", type=ast.literal_eval, default='False', help="If interpolate the rope embeddings while loading from a pretrained model")
parser.add_argument("--if_continue_inf", type=ast.literal_eval, default='True', help="if continue training when inf")
parser.add_argument("--if_nan2num", type=ast.literal_eval, default='True', help="if nan should be converted to num")
parser.add_argument("--if_random_cls_token_position", type=ast.literal_eval, default='False', help="if random cls token position")
parser.add_argument("--if_random_token_rank", type=ast.literal_eval, default='False', help="if random token rank")
parser.add_argument("--aum_drop_path", type=float, default=0, help="aum drop path")
parser.add_argument("--imagenet_load_middle_cls_token", type=ast.literal_eval, default='True', help="If loading from a middle cls token ViM model")
parser.add_argument("--imagenet_load_double_cls_token", type=ast.literal_eval, default='False', help="If loading from a double cls token ViM model")
parser.add_argument("--if_cls_token", type=ast.literal_eval, default='True', help="If use cls token")
parser.add_argument("--use_middle_cls_token", type=ast.literal_eval, default='True', help="If use middle cls token")
parser.add_argument("--use_double_cls_token", type=ast.literal_eval, default='False', help="If use double cls token")
parser.add_argument("--use_end_cls_token", type=ast.literal_eval, default='False', help="If use end cls token")
parser.add_argument("--transpose_token_sequence", type=ast.literal_eval, default='False', help="If transpose token sequence")
parser.add_argument("--aum_type", type=str, default='Fo-Bi', help="aum type of the block (for the AuM)")

# AST
parser.add_argument("--ast_pretrain", type=ast.literal_eval, default='False', help="If initialize with pretrained ast model")
parser.add_argument("--ast_pretrain_path", type=str, default=None, help="pretrain ast model path")
parser.add_argument("--ast_fstride", type=int, default=16, help="pretrain ast model fstride (for positional embeddings)")
parser.add_argument("--ast_tstride", type=int, default=16, help="pretrain ast model tstride (for positional embeddings)")
parser.add_argument("--ast_label_dim", type=int, default=527, help="pretrain ast model label dim")
parser.add_argument("--ast_input_fdim", type=int, default=128, help="pretrain ast model input f dim")
parser.add_argument("--ast_input_tdim", type=int, default=1024, help="pretrain ast model input t dim")
parser.add_argument("--ast_model_name", type=str, default=None, help="pretrain ast model name")
parser.add_argument("--load_backbone_only", type=ast.literal_eval, default='False', help="load backbone only")

# Training
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")
parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")
parser.add_argument("--bs_scale_factor", type=int, default=1, help="batch size scale factor, used to adapt the optimizer params wrt to an originally set batch size")
parser.add_argument("--weight_decay", type=float, default=5e-7, help="weight decay")
parser.add_argument("--optim_path", type=str, default=None, help="optimizer state path (to continue an existing training)")

# Flexible Training
parser.add_argument("--flexible_training", type=ast.literal_eval, default='False', help="flexible training")
parser.add_argument("--flexible_p_start", type=int, default=8, help="flexible p start")
parser.add_argument("--flexible_p_end", type=int, default=50, help="flexible p end")
parser.add_argument("--flexible_p_step", type=int, default=2, help="flexible p step")

args = parser.parse_args()

if args.flexible_training:
    args.flexible_patch_sizes = list(range(args.flexible_p_start, args.flexible_p_end , args.flexible_p_step))
else:
    args.flexible_patch_sizes = []

if args.dataset == 'epic_sounds':
    config_yaml = '../../src/epic_sounds/epic_data/config_default.yaml'
    with open(config_yaml, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = CfgNode(cfg_dict)

    # overwrite parts of the config with the arguments
    cfg.T_MASK = int(args.timem * args.audio_length/1024)
    cfg.F_MASK = args.freqm
    cfg.TEST.BATCH_SIZE = args.batch_size * 2
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.AUDIO_DATA.CLIP_SECS=int(args.audio_length/100)
    cfg.AUDIO_DATA.NUM_FRAMES = args.audio_length
    cfg.T_WARP = 5
    cfg.DATA_LOADER.NUM_WORKERS = args.num_workers

    if args.run_type == 'train':
        train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
else:
    audio_conf = {
        'num_mel_bins': args.melbins, 'target_length': args.audio_length, 'freqm': args.freqm, 
        'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 
        'mode': 'train', 'mean': args.dataset_mean, 'std': args.dataset_std,
        'noise': args.noise, 'fshift': args.fshift
    }
    val_audio_conf = {
        'num_mel_bins': args.melbins, 'target_length': args.audio_length, 'freqm': 0, 
        'timem': 0, 'mixup': 0, 'dataset': args.dataset, 
        'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 
        'noise': False, 'fshift': args.fshift
    }

    if args.run_type == 'train':
        if args.bal == 'bal':
            accelerator.print('balanced sampler is being used')
            samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

            train_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
                batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True
            )
        else:
            accelerator.print('balanced sampler is not used')
            
            train_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
            )

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

# transformer based model
if args.model == 'ast':
    accelerator.print('Model: audio spectrogram transformer (AST)')

    model_name = 'deit'
    model_size = args.model_type.split('_')[0]
    model_name += f'_{model_size}'
    if 'dist' in args.model_type:
        model_name += '_distilled'
    model_name += '_patch16'
    if '384' in args.model_type:
        model_name += '_384'
    else:
        model_name += '_224'

    accelerator.print(f'Now train an AST with the architecture from {model_name} model!')

    audio_model = models.ASTModel(
        label_dim=args.n_class, fstride=args.fstride, 
        tstride=args.tstride, input_fdim=args.melbins, 
        input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
        model_name=model_name, ast_pretrain=args.ast_pretrain, 
        ast_pretrain_path=args.ast_pretrain_path, ast_fstride=args.ast_fstride, 
        ast_tstride=args.ast_tstride, ast_label_dim=args.ast_label_dim,
        ast_input_fdim=args.ast_input_fdim, ast_input_tdim=args.ast_input_tdim,
        ast_model_name=args.ast_model_name, load_backbone_only=args.load_backbone_only
    )

if args.model == 'aum':
    accelerator.print('Model: AudioMamba (AUM)')

    if 'base' in args.model_type:
        depth=24
        embed_dim=768
    elif 'small' in args.model_type:
        depth=24
        embed_dim=384
    elif 'tiny' in args.model_type:
        depth=24
        embed_dim=192
    else:
        raise ValueError('unknown model type, model type should be one of [base, small, tiny] for aum')

    if args.aum_type == 'Fo-Fo':
        bimamba_type = 'none'
    elif args.aum_type == 'Fo-Bi':
        bimamba_type = 'v1'
    elif args.aum_type == 'Bi-Bi':
        bimamba_type = 'v2'
    else:
        raise ValueError('unknown aum type, aum type should be one of [Fo-Fo, Fo-Bi, Bi-Bi] for aum')

    audio_model = models.AudioMamba(
        spectrogram_size=(args.melbins, args.audio_length),
        patch_size=(args.fpatch_size, args.tpatch_size),
        strides=(args.fstride, args.tstride),
        depth=depth,
        embed_dim=embed_dim,
        num_classes=args.n_class,
        imagenet_pretrain=args.imagenet_pretrain,
        imagenet_pretrain_path=args.imagenet_pretrain_path,
        imagenet_pretrain_modelkey=args.imagenet_pretrain_modelkey,
        aum_pretrain=args.aum_pretrain,
        aum_pretrain_path=args.aum_pretrain_path,
        aum_pretrain_fstride=args.aum_pretrain_fstride,
        aum_pretrain_tstride=args.aum_pretrain_tstride,
        pt_hw_seq_len=(args.pt_seq_lenf, args.pt_seq_lent) if args.pt_seq_lenf is not None and args.pt_seq_lent is not None else None,
        bilinear_rope=args.bilinear_rope,
        drop_path_rate=args.aum_drop_path,
        imagenet_load_double_cls_token=args.imagenet_load_double_cls_token,
        imagenet_load_middle_cls_token=args.imagenet_load_middle_cls_token,
        use_double_cls_token=args.use_double_cls_token,
        use_middle_cls_token=args.use_middle_cls_token,
        use_end_cls_token=args.use_end_cls_token,
        bimamba_type=bimamba_type,
        transpose_token_sequence=args.transpose_token_sequence,
        if_cls_token=args.if_cls_token,
        flexible_patch_sizes=args.flexible_patch_sizes
    )

if args.run_type == 'train':

    accelerator.print("\nCreating experiment directory: %s" % args.exp_dir)
    if accelerator.is_main_process:
        os.makedirs("%s/models" % args.exp_dir)
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)

    accelerator.print('Now starting training for {:d} epochs'.format(args.n_epochs))

    args.accelerator = accelerator
    train(audio_model, train_loader, val_loader, args)

elif args.run_type == 'eval':

    audio_model, val_loader = accelerator.prepare(audio_model, val_loader)
    accelerator.print(f'Now starting evaluation on {args.dataset} dataset!')
    args.accelerator = accelerator

    stats, loss = validate_acc(audio_model, val_loader, args, 'eval')
    
    if accelerator.is_main_process:
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if args.metrics == 'mAP':
            accelerator.print("mAP: {:.6f}".format(mAP))
        else:
            accelerator.print("acc: {:.6f}".format(acc))

        accelerator.print("AUC: {:.6f}".format(mAUC))
        accelerator.print("Avg Precision: {:.6f}".format(average_precision))
        accelerator.print("Avg Recall: {:.6f}".format(average_recall))
        accelerator.print("d_prime: {:.6f}".format(d_prime(mAUC)))
        accelerator.print("valid_loss: {:.6f}".format(loss))

        if args.metrics == 'mAP':
            result = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss]
        else:
            result = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss]
    
        np.savetxt(args.exp_dir + f'/result_{str(args.run_type)}.csv', result, delimiter=',')

        accelerator.print('validation finished')

        with open(args.exp_dir + '/stats_' + str(args.run_type) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    raise ValueError('Unknown run type, run type should be one of [train, eval]')

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.run_type == 'train' and args.dataset == 'speechcommands':
    
    accelerator.wait_for_everyone()

    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth')
    
    # remove module from the keys
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    # remove unused keys that has v.head in it (for AST)
    sd = {k: v for k, v in sd.items() if 'v.head' not in k}

    audio_model.load_state_dict(sd)
    val_loader = accelerator.prepare(val_loader)

    # best model on the validation set
    stats, _ = validate_acc(audio_model, val_loader, args, 'valid_set')
    if accelerator.is_main_process:
        # note it is NOT mean of class-wise accuracy
        val_acc = stats[0]['acc']
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the validation set---------------')
        print("Accuracy: {:.6f}".format(val_acc))
        print("AUC: {:.6f}".format(val_mAUC))

    accelerator.wait_for_everyone()

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    eval_loader = accelerator.prepare(eval_loader)
    
    stats, _ = validate_acc(audio_model, eval_loader, args, 'eval_set')
    
    if accelerator.is_main_process:
        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the test set---------------')
        print("Accuracy: {:.6f}".format(eval_acc))
        print("AUC: {:.6f}".format(eval_mAUC))
        np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
