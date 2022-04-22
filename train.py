import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from augmentation.augmented_dataset import AugmentedDataset
from models import TVModel, LFHFModel
import json
import pandas as pd
import copy

os.environ["PYTHONWARNINGS"] = 'ignore:semaphore_tracker:UserWarning'

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names ,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--auto-resume', action='store_true',
                    help='auto resume from the last save checkpoint in save dir')
parser.add_argument('--finetune', action='store_true',
                    help='finetune model')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local node rank for distributed training')
parser.add_argument('--dist-url', default='dummy', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-mp', '--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./work_dir',
    help='Folder to save checkpoints.')

parser.add_argument(
    '--save_pred',
    type=str,
    default=None,
    help="path to save pred")


parser.add_argument('--id', type=str, default='tmp-in', help='An experiment id')


parser.add_argument(
    '--num-classes',
    default=1000,
    type=int,
    help='number of classes')

parser.add_argument(
            '--mixture-width',
            default=3,
            type=int,
            help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')

parser.add_argument(
    '--aug-prob-coeff',
    default=1.,
    type=float,
    help='Probability distribution coefficients')

parser.add_argument(
    '--aug-type',
    choices = ['noaug', 'rand', 'augmix'],
    default='augmix',
    help='Choose aug type')


parser.add_argument(
    '--aug-list',
    nargs="+",
    help='Provide a list of augmentations to use')

parser.add_argument(
    '--imsize',
    default=224,
    type=int,
    help='The training image resolution')


# TV args:
parser.add_argument(
    '--tv',
    action='store_true',
    help='wrap a TV model')

parser.add_argument(
    '--tv-wt',
    default=0.01,
    type=float,
    help='scaling parameter for the TV loss')

parser.add_argument(
    '--num-tv-layers',
    default=1,
    type=int,
    help='the number of TV layers')

parser.add_argument(
    '--tv-layer',
    type=str,
    help='Name of the layer to apply TV')



parser.add_argument(
    '--low-high',
    action='store_true',
    help='use low and high frequency specialist models')

parser.add_argument('--lf-ckpt', default='', type=str,
                    help='path to lf ckpt (default: none)')

parser.add_argument('--hf-ckpt', default='', type=str,
                    help='path to hf ckpt (default: none)')

parser.add_argument(
    '--num-bn-updates',
    '-nb',
    default=0,
    type=int,
    help='number bn update steps at eval (unsupervised domain adaptation)')


# global
best_acc1 = 0

def set_seed(args):
    seed = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_or_create_save_path(args):
    save_path = args.save
    if args.id:
        save_path = os.path.join(args.save, args.id)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def save_train_args(save_path, args):
    if not args.evaluate:
        with open(os.path.join(save_path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

def save_results(results, save_path, epoch):
    print("=> Saving results: ")
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.transpose()
    fname = os.path.join(save_path, "results.csv")
    df.to_csv(fname)


def main():
    args = parser.parse_args()
    save_path = get_or_create_save_path(args)
    save_train_args(save_path, args)


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed or args.dist_url == "env://"

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        print("=> Using spawn method")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("=> Calling worker")
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def load_checkpoint(model, optimizer, args):
    ckpt_path = args.resume
    if os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        if args.gpu is None:
            checkpoint = torch.load(ckpt_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(ckpt_path, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        if not args.finetune:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            if args.tv:
                new_state_dict = {}
                for k,v in checkpoint['state_dict'].items():
                    new_key  = k.replace("module.", "").replace("model.", "")
                    new_state_dict[new_key] = v
                model.module.model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['state_dict'])

            print("=> Finetuning mode (reseting best acc and start epoch)")
            args.start_epoch = 0
            best_acc1 = 0
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(ckpt_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))
        if args.start_epoch!=0:
            raise(FileNotFoundError("No ckpt to resume!"))


def main_worker(gpu, ngpus_per_node, args):
    set_seed(args)
    global best_acc1

    if args.dist_url == "env://" and args.local_rank == -1:
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        args.gpu = gpu

    if args.gpu is not None:
        print("Worker -> using GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    kwargs = {}

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, **kwargs)
    else:
        print("=> creating model '{}'".format(args.arch))
        print("=> num classes in model: {}".format(args.num_classes))
        if args.tv:
            model = models.__dict__[args.arch](num_classes=args.num_classes, **kwargs)
            print("=> adding tv model wrappers")
            model = TVModel(model, args.num_tv_layers, layer_name=args.tv_layer)
        elif args.low_high:
            print("=> adding LF HF wrappers")
            lf_model = models.__dict__[args.arch](num_classes=args.num_classes, **kwargs)
            hf_model = models.__dict__[args.arch](num_classes=args.num_classes, **kwargs)
            hf_model = TVModel(hf_model, args.num_tv_layers)
            weight_predictor=None
            model = LFHFModel(lf_model, hf_model, args.lf_ckpt, args.hf_ckpt)
        else:
            print("=> standard model")
            model = models.__dict__[args.arch](num_classes=args.num_classes, **kwargs)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            print("DataParallel")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)


    save_path = get_or_create_save_path(args)

    # optionally resume from a checkpoint
    if args.resume:
        load_checkpoint(model, optimizer, args)
    elif args.auto_resume:
        print("=> Executing auto resume")
        ckpt_path = os.path.join(save_path, "checkpoint.pth.tar")
        args.resume = ckpt_path
        #args.finetune = False
        load_checkpoint(model, optimizer, args)


    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([transforms.ToTensor(),
                                     normalize])

    train_transform = transforms.Compose(
      [transforms.RandomResizedCrop(args.imsize),
       transforms.RandomHorizontalFlip(),
       ])

    test_transform = transforms.Compose([
      transforms.Resize(int(256*(args.imsize/224))),
      transforms.CenterCrop(args.imsize),
      preprocess])

    train_dataset = datasets.ImageFolder( traindir,
                                          train_transform)

    if args.aug_type == 'noaug':
        args.no_jsd = True

    train_dataset = AugmentedDataset(train_dataset, preprocess, args)

    if args.distributed:
        print("Using distributed sampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, test_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        results_clean = {}
        pred_dump = os.path.join(save_path, "preds")
        #os.makedirs(pred_dump, exist_ok=True)
        args.save_pred = None
        acc, loss = validate(val_loader, model, criterion, args)
        results_clean['clean_acc'] = [float(acc)]
        results_clean['clean_loss'] = [float(loss)]
        results_corr = test_c(model, preprocess, criterion, args)
        results = {**results_clean, **results_corr}
        save_results(results, save_path, args.start_epoch)
        return

    log_path = os.path.join(save_path,
                          'imagenet_{}_training_log.csv'.format(args.arch))
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,batch_time,train_loss,train_acc1(%),test_loss,test_acc1(%)\n')

    if args.distributed and args.start_epoch==0 and (not args.finetune):
        wepochs = 5
        print("Warming up...({} epochs)".format(wepochs))
        for epoch in range(0, wepochs):
            train_sampler.set_epoch(epoch)
            lr = args.lr * (1./(wepochs-epoch))
            print("Warmup Lr @ epoch {}: {:.5f}".format(epoch, lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if args.wt_predictor:
                train_acc1, train_loss, batch_time = train_wp(train_loader, model, criterion, optimizer, epoch, args)
            else:
                train_acc1, train_loss, batch_time = train(train_loader, model, criterion, optimizer, epoch, args)

            if (args.gpu == 0 and args.rank == 0):
                save_checkpoint({
                'epoch': 0,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                },
                False,
                os.path.join(save_path, "checkpoint.pth.tar")
            )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc1, train_loss, batch_time = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        test_acc1, test_loss = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        if not args.distributed or (args.distributed and args.gpu == 0 and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            },
            is_best,
            os.path.join(save_path, "checkpoint.pth.tar")
            )
            with open(log_path, 'a') as f:
                f.write('%03d,%0.3f,%0.6f,%0.2f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                batch_time,
                train_loss,
                train_acc1,
                test_loss,
                test_acc1))



def jsd_loss(logits_clean, logits_aug1, logits_aug2):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)
    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    return loss

def compute_tv_loss(model, args):
    if args.tv and args.tv_wt > 0:
        tv_losses = model.module.tv_losses
        if args.num_tv_layers:
            assert(len(tv_losses) == args.num_tv_layers)
        tv_loss = sum(tv_losses)/1000
        tv_loss = tv_loss*args.tv_wt
    else:
        tv_loss = torch.tensor(0.)
    return tv_loss


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    tv_losses = AverageMeter('TvLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, tv_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)

        if args.no_jsd:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
        else:
            images_all = torch.cat(images, 0)
            if args.gpu is not None:
                images_all = images_all.cuda(args.gpu, non_blocking=True)

            images_all = images_all.cuda(args.gpu, non_blocking=True)
            logits_all = model(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = criterion(logits_clean, targets)
            loss += jsd_loss(logits_clean, logits_aug1, logits_aug2)
            logits = logits_clean

        tv_loss = compute_tv_loss(model, args)
        loss += tv_loss
        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        size = None
        if isinstance(images, list):
            size = images[0].size(0)
        else:
            size = images.size(0)

        losses.update(loss.item(), size)
        tv_losses.update(tv_loss.item(), size)

        top1.update(acc1[0], size)
        top5.update(acc5[0], size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, losses.avg, batch_time.avg

def test_c(model, test_transform, criterion, args):
    corr_results = {}
    corr_data = os.path.join(args.data, 'corrupted')
    for c in CORRUPTIONS:
        print("Evaluating: {}".format(c))
        for s in range(1, 6):
            valdir = os.path.join(corr_data, c, str(s))
            val_loader = torch.utils.data.DataLoader(
                                                    datasets.ImageFolder(valdir, test_transform),
                                                    batch_size=args.batch_size,
                                                    shuffle=False, num_workers=args.workers,)
            if args.save_pred:
                basename = os.path.basename(args.save_pred)
                args.save_pred = args.save_pred.replace(basename, "{}_{}".format(c,s))
            acc1, loss = validate(val_loader, model, criterion, args)
            acc_key = "{}_acc".format(c)
            loss_key = "{}_loss".format(c)
            if not acc_key in corr_results:
                corr_results[acc_key] = [float(acc1)]
                corr_results[loss_key] = [float(loss)]
            else:
                corr_results[acc_key].append(float(acc1))
                corr_results[loss_key].append(float(loss))
            print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(s, loss, acc1))
    return corr_results


def update_bn_params(model, val_loader, args):
    val_loader = torch.utils.data.DataLoader(val_loader.dataset,
                                             batch_size=val_loader.batch_size,
                                             shuffle=True, num_workers=val_loader.num_workers)
    def use_test_statistics(module):
        if isinstance(module, nn.BatchNorm2d):
            module.train()
    model = copy.deepcopy(model)
    model.eval()
    model.apply(use_test_statistics)
    print("Updating BN params (num updates:{})".format(args.num_bn_updates))
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i<args.num_bn_updates:
                images = images.cuda(args.gpu, non_blocking=True)
                output = model(images)
    print("Done.")
    return model



def validate(val_loader, model, criterion, args, mask=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    if args.num_bn_updates>0:
        model = update_bn_params(model, val_loader, args)
    # switch to evaluate mode
    model.eval()
    predictions = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            if not mask is None:
                output = output[:, mask]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

            if args.save_pred:
                prob = F.softmax(output, dim=1).cpu().numpy()
                predictions.append(prob)

        if args.save_pred:
            pred = np.concatenate(predictions, axis=0)
            np.save(args.save_pred, pred)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    dirname = os.path.dirname(filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    b = 1
    k = args.epochs // 3
    if epoch < k:
        m = 1
    elif epoch < 2 * k:
        m = 0.1
    else:
        m = 0.01
    lr = args.lr * m * b
    print("Lr @ epoch {}: {:.5f}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

if __name__ == '__main__':
    main()
