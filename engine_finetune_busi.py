# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from collections import defaultdict


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    TP, TN, FP, FN = 0, 0, 0, 0
    TP_slice, TN_slice, FP_slice, FN_slice = 0, 0, 0, 0

    dict_correct = defaultdict(int)
    dict_total = defaultdict(int)
    dict_label = defaultdict(int)
    dict_pred = defaultdict(int)
    ss_target = []
    ss_pred = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        
        # cases = [single_case.split('/')[-1].split('_')[0].split('case')[-1] for single_case in batch[2]]
        # cases = [''.join(single_case.split('/')[-1].split('-')[:-1]) for single_case in batch[2]]
        cases = [single_case.split('/')[-1] for single_case in batch[2]]

        # cases_x = batch[2]

        # print(cases)
        
        for i in range(len(cases)):
            dict_label[cases[i]] = target[i]
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # print("Target-> ", target)
            ss = torch.argmax(output, dim=1)
            # print("output->", torch.argmax(output, dim=1))
            ss_pred.extend(ss.detach().cpu().numpy())
            ss_target.extend(target.detach().cpu().numpy())
            
            for i in range(len(cases)):
                dict_total[cases[i]] += 1
                print("case -> ", cases[i])
                # print(target[i], cases_x[i])

                if((target[i] == 0 or target[i] == 2) and (ss[i]==2 or ss[i]==0)):
                    dict_correct[cases[i]] += 1 
                    # print("xxxxx ", target[i], ss[i])
                    TN_slice += 1
                elif(target[i] == 1 and ss[i]==1):
                    # print("xxxxx ", target[i], ss[i])
                    dict_correct[cases[i]] += 1
                    TP_slice += 1
                elif((target[i] == 0 or target[i] == 2) and ss[i] == 1):
                    # print("xxxxx ", target[i], ss[i])
                    FP_slice += 1
                elif(target[i] == 1 and (ss[i] ==0 or ss[i]==2)):
                    # print("xxxxx ", target[i], ss[i])
                    FN_slice +=1 
                else:
                    print(target[i], ss[i])
                    print("SOMETHING WENT HORRIBLY WRONG", target[i], ss[i])

                    
                    

            loss = criterion(output, target)
            
        

        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(ss_pred, ss_target)
    # print("wefffffffffffffffff", confusion_matrix(ss_target, ss_pred))
    threshold = 0.10
    # for cases in dict_total:
    #     if(dict_correct[cases] / dict_total[cases] >= 1 - threshold and (dict_label[cases] == 0 or dict_label[cases] == 2)):
    #         TN += 1
    #     elif(dict_correct[cases] / dict_total[cases] >= threshold and dict_label[cases] == 1):
    #         TP += 1
    #     elif(dict_correct[cases] / dict_total[cases] < 1 - threshold and dict_label[cases] == 1):
    #         FN += 1
    #     else :
    #         FP += 1
    # cm = [[TN, FP], [FN, TP]]
    cm_slice = [[TN_slice, FP_slice], [FN_slice, TP_slice]]

    
    # sensitivity = TP / float(TP + FN)
    # specificity = TN / float(TN + FP)
    
    sensitivity_slice = TP_slice / float(TP_slice + FN_slice)
    specificity_slice = TN_slice / float(TN_slice + FP_slice)
    # print("Confusion Matix case wise-> ", cm)
    # print("Specificity Case -> ", specificity)
    # print("Sensitivity Case -> ", sensitivity)
    # print("Acc(case) -> ", (TP + TN) / (TP + TN + FN + FP))

    
    print("Confusion Matix slice wise-> ", cm_slice)
    print("Specificity Slice-> ", specificity_slice)
    print("Sensitivity Slice-> ", sensitivity_slice)
    print("Acc(silce) -> ", (TP_slice + TN_slice) / (TP_slice + TN_slice + FN_slice + FP_slice))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}