import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm
import torch.distributed as dist

def train(audio_model, train_loader, test_loader, args):
    
    accelerator = args.accelerator

    torch.set_grad_enabled(True)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]

    accelerator.print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    accelerator.print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    
    def scale_beta(beta):
        return 1 - (1 - beta) * args.bs_scale_factor

    def scale_eps(eps):
        return eps / (args.bs_scale_factor ** 0.5)

    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=args.weight_decay, 
                                 betas=(scale_beta(0.95), scale_beta(0.999)), 
                                 eps=scale_eps(1e-8))
    
    if args.optim_path:
        optimizer.load_state_dict(torch.load(args.optim_path))
    
    audio_model, train_loader, test_loader, optimizer = accelerator.prepare(audio_model, train_loader, test_loader, optimizer)

    if accelerator.is_main_process:
        loss_meter = AverageMeter()
        progress = []

        best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
        exp_dir = args.exp_dir

        def _save_progress():
            progress.append([epoch, global_step, best_epoch, best_mAP, best_acc])
            with open("%s/progress.pkl" % exp_dir, "wb") as f:
                pickle.dump(progress, f)
        
        result = np.zeros([args.n_epochs, 8])

    main_metrics = args.metrics
    
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    warmup = args.warmup
    
    if args.dataset == 'epic_sounds':
        def epic_lr_schedule(epoch):
            if epoch < 10:
                return 1.0  # No decay for epochs < 10
            elif epoch < 20:
                return 0.05  # Decays to 5% of the initial LR for epochs 10-19
            else:
                return 0.01  # Decays to 1% of the initial LR for epochs >= 20
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=epic_lr_schedule)
        epic_warmup_step = 2 * len(train_loader)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)

    accelerator.print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    accelerator.print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    
    global_step, epoch = 0, 0
    epoch += 1
    
    accelerator.print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    accelerator.print("start training...")
    
    while epoch < args.n_epochs + 1:
        audio_model.train()
    
        accelerator.print('---------------')
        accelerator.print(datetime.datetime.now())
        accelerator.print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        if accelerator.is_main_process:
            tepoch = tqdm(train_loader)
        else:
            tepoch = train_loader

        for i, batch in enumerate(tepoch):
            
            if args.dataset == 'epic_sounds':
                audio_input, labels, _, _ = batch
            else:
                audio_input, labels, path = batch

            B = audio_input.size(0)

            if args.dataset == 'epic_sounds':
                if global_step < epic_warmup_step and warmup == True:
                    warm_lr = args.lr * 0.01 + global_step * (args.lr - args.lr * 0.01) / epic_warmup_step
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warm_lr
                    if global_step % 100 == 0:
                        accelerator.print(f'warm-up learning rate is {warm_lr}')
                elif global_step >= epic_warmup_step and warmup == True:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
                    if global_step == epic_warmup_step:    
                        accelerator.print('end of warm-up, learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))
            else:
            # first several steps for warm-up
                if global_step <= (1000 // args.bs_scale_factor) and global_step % (50 // args.bs_scale_factor) == 0 and warmup == True:
                    warm_lr = (global_step / (1000 // args.bs_scale_factor)) * args.lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warm_lr
                    accelerator.print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            if args.model == 'aum':
                if args.flexible_training:
                    if accelerator.is_main_process:
                        patch_size = random.choice(args.flexible_patch_sizes)
                        strides = patch_size # TODO: consider flexifying the strides too
                    else:
                        patch_size, strides = -1, -1 # arbitrary value that will be overwritten
                    # tensorize
                    patch_size = torch.tensor(patch_size).to(accelerator.device)
                    strides = torch.tensor(strides).to(accelerator.device)
                    # broadcast the patch_size and strides to all processes
                    dist.broadcast(patch_size, src=0)
                    dist.broadcast(strides, src=0)
                    # itemize
                    patch_size = patch_size.item()
                    strides = strides.item()
                else:
                    patch_size, strides = None, None # sets to default patch_size and strides
                audio_output = audio_model(audio_input, if_random_cls_token_position=args.if_random_cls_token_position, patch_size=patch_size, strides=strides)
            else:
                # TODO: not yet implemented flexible training for AST
                audio_output = audio_model(audio_input)
            
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = loss_fn(audio_output, labels)
            
            if args.if_nan2num:
                loss = torch.nan_to_num(loss)

            loss_value = loss.item()
            if not math.isfinite(loss_value): # TODO: Handle this in accelerator case
                if args.if_continue_inf and accelerator.is_main_process:
                    print("Loss is {}, continuing training".format(loss_value))
                    optimizer.zero_grad()
                    continue
                else:
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            loss_tensor = torch.tensor([loss_value]).to(accelerator.device)
            bs_tensor = torch.tensor([B]).to(accelerator.device)
            gathered_loss_tensor = accelerator.gather(loss_tensor)
            gathered_bs_tensor = accelerator.gather(bs_tensor)

            if accelerator.is_main_process:
                for i in range(len(gathered_loss_tensor)):
                    loss_meter.update(gathered_loss_tensor[i].item(), gathered_bs_tensor[i].item())
                bar_dict = {"Epoch":epoch,"T_Loss":loss_meter.avg} 
                tepoch.set_postfix(bar_dict)
            
            global_step += 1

        
        accelerator.print('start validation')
    
        args.accelerator = accelerator
        stats, valid_loss = validate_acc(audio_model, test_loader, args, epoch)

        if accelerator.is_main_process:
            mAP = np.mean([stat['AP'] for stat in stats])
            mAUC = np.mean([stat['auc'] for stat in stats])
            acc = stats[0]['acc']

            middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
            middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
            average_precision = np.mean(middle_ps)
            average_recall = np.mean(middle_rs)

            if main_metrics == 'mAP':
                accelerator.print("mAP: {:.6f}".format(mAP))
            else:
                accelerator.print("acc: {:.6f}".format(acc))
            accelerator.print("AUC: {:.6f}".format(mAUC))
            accelerator.print("Avg Precision: {:.6f}".format(average_precision))
            accelerator.print("Avg Recall: {:.6f}".format(average_recall))
            accelerator.print("d_prime: {:.6f}".format(d_prime(mAUC)))
            accelerator.print("train_loss: {:.6f}".format(loss_meter.avg))
            accelerator.print("valid_loss: {:.6f}".format(valid_loss))

            if main_metrics == 'mAP':
                result[epoch-1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, optimizer.param_groups[0]['lr']]
            else:
                result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, optimizer.param_groups[0]['lr']]
        
            np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
    
            accelerator.print('validation finished')

            if mAP > best_mAP:
                best_mAP = mAP
                if main_metrics == 'mAP':
                    best_epoch = epoch

            if acc > best_acc:
                best_acc = acc
                if main_metrics == 'acc':
                    best_epoch = epoch

            if best_epoch == epoch:
                torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
                torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))


            torch.save(audio_model.state_dict(), "%s/models/latest_audio_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/latest_optim_state.%d.pth" % (exp_dir, epoch))
        
            accelerator.print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

            with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
                pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            _save_progress()

            loss_meter.reset()
        
        accelerator.wait_for_everyone()
        scheduler.step()
        epoch += 1

def validate_acc(audio_model, val_loader, args, epoch, save_pred=True):
    
    accelerator = args.accelerator

    audio_model.eval()

    if accelerator.is_main_process:
        A_predictions = []
        A_targets = []
        A_loss = []

    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError('loss function not defined')

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if args.dataset == 'epic_sounds':
                audio_input, labels, _, _ = batch
            else:
                audio_input, labels, path = batch
            audio_output = audio_model(audio_input)
            if args.if_nan2num:
                audio_output = torch.nan_to_num(audio_output)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.detach()

            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = loss_fn(audio_output, labels)

            gathered_predictions = accelerator.gather(predictions)
            gathered_labels = accelerator.gather(labels)
            gathered_loss = accelerator.gather(loss)

            if len(gathered_loss.shape) == 0:
                gathered_loss = gathered_loss.unsqueeze(0)

            if accelerator.is_main_process:
                A_predictions.append(gathered_predictions)
                A_targets.append(gathered_labels)
                A_loss.append(gathered_loss)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            audio_output = torch.cat(A_predictions)
            audio_output = audio_output.to('cpu').detach()
            
            target = torch.cat(A_targets)
            target = target.to('cpu').detach()
            
            loss = torch.cat(A_loss)
            loss = torch.mean(loss).item()
            
            stats = calculate_stats(audio_output, target)

            if save_pred:
                exp_dir = args.exp_dir
                if os.path.exists(exp_dir+'/predictions') == False:
                    os.mkdir(exp_dir+'/predictions')
                    np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
                np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')
        else: 
            stats = None
            loss = None

    return stats, loss        

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats

# Note: this function has not been used and tested, may have bugs, kept here as a reference. 
def validate_wa(audio_model, val_loader, accelerator, args, start_epoch, end_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    sdA = torch.load(exp_dir + '/models/latest_audio_model.' + str(start_epoch) + '.pth', map_location=device)

    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/latest_audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
        if args.save_model == False:
            os.remove(exp_dir + '/models/latest_audio_model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    print(audio_model.load_state_dict(sdA, strict=False))

    torch.save(audio_model.state_dict(), exp_dir + '/models/latest_audio_model_wa.pth')
    
    audio_model, val_loader = accelerator.prepare(audio_model, val_loader)
    
    stats, loss = validate_acc(audio_model, val_loader, args, 'wa')
    return stats