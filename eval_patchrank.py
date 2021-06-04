import os
import argparse
import random
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import default_cfgs as timm_vit_cfgs
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from patch_game.builder import PatchGame

import utils


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


# TODO: Move these 2 functions to a submodule of timm inside our project as this is a very dirty way of overriding
#  class methods
def forward_features(self, x, patch_inds=None):
    B = x.shape[0]
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.pos_embed
    if patch_inds is not None:
        cls_inds = torch.ones(patch_inds.size(0), 1).to(patch_inds.device)
        patch_inds = torch.cat([cls_inds, patch_inds], dim=1)
        patch_inds = patch_inds.bool()
        patch_inds = patch_inds.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        x = x[patch_inds]
        x = x.reshape(B, -1, self.embed_dim)
    x = self.pos_drop(x)

    x = self.blocks(x)
    x = self.norm(x)
    if self.dist_token is None:
        return self.pre_logits(x[:, 0])
    else:
        return x[:, 0], x[:, 1]


def forward(self, x, patch_inds=None):
    x = self.forward_features(x, patch_inds)
    if self.head_dist is not None:
        x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        if self.training and not torch.jit.is_scripting():
            # during inference, return the average of both classifier predictions
            return x, x_dist
        else:
            return (x + x_dist) / 2
    else:
        x = self.head(x)
    return x


def eval(args):
    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    vit_model = timm.create_model(args.vit_model, pretrained=True)
    vit_model.forward_features = forward_features.__get__(vit_model, VisionTransformer)
    vit_model.forward = forward.__get__(vit_model, VisionTransformer)
    vit_model.cuda()
    vit_model = nn.SyncBatchNorm.convert_sync_batchnorm(vit_model)
    vit_model = nn.parallel.DistributedDataParallel(vit_model, device_ids=[args.gpu])
    vit_model = vit_model.eval()
    patch_model = PatchGame(patch_size=args.patch_size, sender_arch=args.sender_arch,
            sender_hidden_size=args.patch_hidden_size, sender_dropout=args.sender_dropout,
            use_context=args.use_context, sender_norm=args.sender_norm,
            vocab_size=args.vocab_size, max_len=args.max_len, temperature=args.temperature,
            trainable_temperature=args.trainable_temperature, hard=args.hard,
            receiver_arch=args.receiver_arch, receiver_dim=args.receiver_dim,
            receiver_hidden_size=args.receiver_hidden_size,
            num_heads=args.receiver_num_heads, num_layers=args.receiver_num_layers,
            topk=args.topk
        )
    patch_model = patch_model.cuda()
    patch_model = nn.parallel.DistributedDataParallel(patch_model, device_ids=[args.gpu])
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        args.patch_model,
        run_variables=to_restore,
        model=patch_model,
    )
    patch_model = patch_model.eval()

    print('Computing Train Accuracy')
    train_accuracy = eval_patches(data_loader_train, vit_model, patch_model)
    print('Computing Val Accuracy')
    val_accuracy = eval_patches(data_loader_val, vit_model, patch_model, args)


@torch.no_grad()
def eval_patches(loader, vit_model, patch_model, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for inp, target in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = nn.functional.interpolate(inp, (args.im_size, args.im_size))
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            _, patches = patch_model.module.sender(inp)
            output = vit_model(inp, patches)
        loss = nn.CrossEntropyLoss()(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with pretrained ViT')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--vit_model', default='vit_base_patch32_384', type=str,
                        help='Pretrained ViT from timm.')
    parser.add_argument('--speaker_arch', default='resnet', type=str,
        help='Architecture of speaker')
    parser.add_argument('--patch_model', default='', type=str, help="Path to pretrained weights for speaker")
    parser.add_argument('--im_size', default=384, type=int, help='Image size to be used')
    parser.add_argument('--patch_size', default=32, type=int, help='Patch resolution of the model.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=1, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

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

    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    args = parser.parse_args()

    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Checking if selected pretrained vit is compatible with other options
    assert args.vit_model in timm_vit_cfgs.keys(), f'vit_model:{args.vit_model} is not available in timm'
    vit_cfg = timm_vit_cfgs[args.vit_model]
    assert vit_cfg['input_size'] == (3, args.im_size, args.im_size), f'vit_model:{args.vit_model} has a size that is not supported'
    vit_patch_size = int(args.vit_model.split('_')[-2].replace('patch', ''))
    assert vit_patch_size == args.patch_size, f'vit_model={args.vit_model} has a patch_size={vit_patch_size} but ' \
                                              f'given rank model has patch_size={args.patch_size}'
    eval(args)


