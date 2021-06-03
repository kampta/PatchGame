"""
Training script

Mostly copy-paste from torchvision references or other public repos such as:
https://github.com/facebookresearch/detr,
https://github.com/facebookresearch/deit,
https://github.com/facebookresearch/barlowtwins
"""

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms

import utils
import patch_game.builder as builder


def get_args_parser():
    parser = argparse.ArgumentParser('PatchGame', add_help=False)

    # Model parameters
    parser.add_argument('--patch_size', default=32, type=int, help="""Size in pixels
        of input square patches - default 32 (for 32x32 patches). Using smaller
        values leads to better performance but requires more memory.""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger models.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0., help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=30, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Game play
    # Sender
    parser.add_argument('--patch_hidden_size', default=768, type=int)
    parser.add_argument('--sender_arch', default='resnet', choices=['resnet', 'vit_tiny'])
    parser.add_argument('--sender_norm', default='sort')
    parser.add_argument('--use_context', type=utils.bool_flag, default=True)
    parser.add_argument('--sender_dropout', default=0.1, type=float)
    parser.add_argument('--vocab_size', default=128, type=int)
    parser.add_argument('--max_len', default=1, type=int)
    parser.add_argument('--start_temperature', default=5.0, type=float)
    parser.add_argument('--warmup_gumbel_epochs', default=None, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--trainable_temperature', type=utils.bool_flag, default=False)
    parser.add_argument('--hard', type=utils.bool_flag, default=True)
    parser.add_argument('--topk', default=None, type=int)

    # Receiver
    parser.add_argument('--receiver_arch', default='resnet18')
    parser.add_argument('--receiver_dim', default=65536, type=int)
    parser.add_argument('--receiver_hidden_size', default=192, type=int)
    parser.add_argument('--receiver_num_heads', default=3, type=int)
    parser.add_argument('--receiver_num_layers', default=12, type=int)

    return parser


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentation(args.global_crops_scale)
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    model = builder.PatchGame(
            image_size=224, patch_size=args.patch_size, sender_arch=args.sender_arch,
            sender_hidden_size=args.patch_hidden_size, sender_dropout=args.sender_dropout,
            use_context=args.use_context, sender_norm=args.sender_norm, 
            vocab_size=args.vocab_size, max_len=args.max_len, temperature=args.temperature, 
            trainable_temperature=args.trainable_temperature, hard=args.hard,
            receiver_arch=args.receiver_arch, receiver_dim=args.receiver_dim, 
            receiver_hidden_size=args.receiver_hidden_size, 
            num_heads=args.receiver_num_heads, num_layers=args.receiver_num_layers,
            topk=args.topk
        )

    # move networks to gpu
    model = model.cuda()
    # synchronize batch norms (if any)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    print(f"Sender has {args.sender_arch} network. Receiver has {args.receiver_arch} network.")

    criterion = nn.CrossEntropyLoss().cuda()

    # ============ preparing optimizer ... ============
    # params_groups = utils.get_params_groups(student)
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    warmup_gumbel_epochs = args.epochs // 2 if args.warmup_gumbel_epochs is None else args.warmup_gumbel_epochs
    temperature_schedule = utils.temperature_scheduler(args.start_temperature, args.temperature, args.epochs, len(data_loader), warmup_gumbel_epochs)
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting PatchGame training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(model, criterion,
            data_loader, optimizer, lr_schedule, wd_schedule,
            epoch, fp16_scaler, args, temperature_schedule)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, criterion, data_loader,
                    optimizer, lr_schedule, wd_schedule, epoch,
                    fp16_scaler, args, temperature_schedule):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            model_out1, model_out2 = model(images[0], images[1], temperature=temperature_schedule[it])
            metric_logger.update(temp=temperature_schedule[it])
            loss = criterion(model_out1, model_out2)
            acc1, acc5 = accuracy(model_out1, model_out2, topk=(1, 5))
            metric_logger.update(acc1=acc1[0])
            metric_logger.update(acc5=acc5[0])

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class DataAugmentation(object):
    def __init__(self, global_crops_scale):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)