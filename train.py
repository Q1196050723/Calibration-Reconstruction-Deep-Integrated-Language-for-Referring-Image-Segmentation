import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
# import wandb
from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F
import ipdb
import gc
from collections import OrderedDict
from ipdb import set_trace as st

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


# IoU calculation for validation
# def IoU(pred, gt):
#     print('pred:',pred.shape)
#     print('gt:',gt.shape)
#     pred = pred.argmax(1)

#     intersection = torch.sum(torch.mul(pred, gt))
#     union = torch.sum(torch.add(pred, gt)) - intersection

#     if intersection == 0 or union == 0:
#         iou = 0
#     else:
#         iou = float(intersection) / float(union)

#     return iou, intersection, union

def IoU(pred,target):
    pred = torch.sigmoid(pred)
    target = target.unsqueeze(1)
    pred = pred.cpu().numpy()
    if pred.shape[-2:] != target.shape[-2:]:
        
        target = target.float()
        target = F.interpolate(target, pred.shape[-2:],mode='nearest').detach()
    pred = np.array(pred > 0.35)
    target = target.cpu().numpy()
    inter = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(inter) / (np.sum(union) + 1e-6)
    return iou,inter,union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


# def criterion(input, target):
#     print(target.shape)
#     weight = torch.FloatTensor([0.9, 1.1]).cuda()
#     return nn.functional.cross_entropy(input, target, weight=weight)

def criterion(pred, target):
    target = target.unsqueeze(1)
   
    target = target.float()
    if pred.shape[-2:] != target.shape[-2:]:
        
        target = F.interpolate(target, pred.shape[-2:],mode='nearest').detach()
    
    weight = torch.tensor([1.1/0.9]).cuda()
    loss = F.binary_cross_entropy_with_logits(input=pred,target=target,weight=weight)
    # weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return loss


def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                
                last_hidden_states,class_token = bert_model(sentences, attention_mask=attentions)

                
                embedding = last_hidden_states.permute(0, 2, 1)
                 # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output,re_loss = model(image, embedding, class_token,l_mask=attentions)
            else:
                output,re_loss = model(image, sentences,class_token, l_mask=attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    # results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
        
    results_str += '    overall IoU = %.2f\n' % (cum_I.sum()*100./cum_U.sum())
    print(results_str)

    return 100 * iou, cum_I.sum()*100./cum_U.sum()


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)
        
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states,class_token = bert_model(sentences, attention_mask=attentions) 
            # last_hidden_states = last_hidden_states + class_token.unsqueeze(1).expand(4,20,768) # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            
            
            # print('cls:',cls_token.shape) 
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output,re_loss = model(image, embedding, class_token, l_mask=attentions)
        else:
            output,re_loss = model(image, sentences,class_token, l_mask=attentions)
        # print('output1:',output1.shape)
        
        loss = criterion(output, target)+0.1*re_loss
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        # if utils.get_rank() in [-1,0]:
        #     wandb.log(
        #             {
        #             "time/loss": loss.item()

        #             })
        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)
    # if args.local_rank == 0:
    #     wandb.init(project='LAVT_EAVL',
    #                 mode='offline',
    #                 name='Swin-B',
    #                 config=args,
    #                 resume='None',
    #                 tags=['refcoco++'])
    # torch.distributed.barrier()
    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        # bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        # params_to_optimize = [
        #     {'params': backbone_no_decay, 'weight_decay': 0.0},
        #     {'params': backbone_decay},
        #     # the following are the parameters of bert
        #     {"params": reduce(operator.concat,
        #                       [[p for p in single_bert_model.encoder.layer[i].parameters()
        #                         if p.requires_grad] for i in range(10)])},
        # ]
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
