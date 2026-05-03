# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:39:28 2023

@author: fbtek
"""

import argparse
import json
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import SequentialLR, LinearLR,StepLR, OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import Subset
from GaussianGlobalAveragePooling2D import AdaptiveGaussianMasked2DPooling
from GaussianGlobalDynamicPooling2D import AdaptiveGaussianDynamic2DPooling
from AdaptiveLocal2DLayerv2 import AdaptiveLocal2DLayer  # <-- Replace with actual path
from resnet_w_alc import ResNetWithALC_ModifiedStride,ResNetWithALC_14x14, ResNetChopHead14x14,ResNetWithALC_14x14_ChannelSep, ResNetWithCoordConvHead14x14, ResNetWithLocalConnectedHead14x14, ResNetWithDeformConvHead14x14
from vit_w_alc import ViTWithALC, ViTStandardHead
from gaussScheduler import GaussianLR
from torchinfo import summary
from torch.cuda.amp import autocast, GradScaler

import pathlib

import numpy as np
from torch.optim.lr_scheduler import LRScheduler

class GaussLR(LRScheduler):
    def __init__(self, optimizer, center, min_lr=1e-5, max_lr=1e-2, lr_sigma=0.25, last_epoch=-1, verbose=False):
        self.center = center
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_sigma = lr_sigma
        super(GaussLR, self).__init__(optimizer, last_epoch, verbose)
        print("Using Gauss LR scheduler!")

    def get_last_lr(self):
        print("updating LR")
        return [self.min_lr + self.max_lr * np.exp(-(self.last_epoch-self.center)**2 / (self.center*self.lr_sigma)**2)
                for base_lr in self.base_lrs]


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default=r'C:\datasets\ILSVRC\Data\CLS-LOC',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--opt', metavar='OPT', default='sgd',
                    help='choose sgd or adamw (default: sgd)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
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
                    metavar='W', help='weight decay (default  for sgd: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--chophead', dest='chophead',action='store_true', help='replace classfication layers weights')
parser.add_argument('--L1Softmax', default=0, type=float, help="L1Softmax")
parser.add_argument('--gausspool',dest='gausspool', action='store_true',help="gausspool")
parser.add_argument('--dynpool', dest='dynpool', action='store_true', help="dynpool")
parser.add_argument('--alc2d', dest='alc2d', action='store_true', help='adaptive locally connected 2D unit')
parser.add_argument('--resnet-alc-variant', default='channel-separate',
                    choices=['full', 'channel-separate', 'modified-stride'],
                    help='ResNet ALC2D integration variant to use when --alc2d is enabled')
parser.add_argument('--resnet-head', default='',
                    choices=['', 'chophead', 'coordconv', 'localconnected', 'deformconv'],
                    help='explicit ResNet head at the shared 14x14 layer3 interface; overrides --chophead for ResNet')
parser.add_argument('--scheduler', default='step', choices=['step', 'cosine', 'gauss'], help="LR scheduler type")
parser.add_argument('--freeze_base', dest='freeze_base', action='store_true', help="want to freeze weights of base model layers ")
parser.add_argument('--outputpath', default=None, type=pathlib.Path, help='outputs such as checkpoints saved here')
parser.add_argument('--run-name', default='', type=str, help='optional human-readable run name used in output folders and metadata')
parser.add_argument('--log-optimizer-params', dest='log_optimizer_params', action='store_true',
                    help='log every parameter name added to the optimizer')
parser.add_argument('--max-train-batches', default=0, type=int,
                    help='optional limit for training batches per epoch, useful for local smoke tests')
parser.add_argument('--max-val-batches', default=0, type=int,
                    help='optional limit for validation batches, useful for local smoke tests')
best_acc1 = 0


def sanitize_name(value):
    if not value:
        return ''
    sanitized = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value.strip())
    return sanitized.strip('_')


def resolve_output_dir(args):
    if args.outputpath is not None:
        output_dir = pathlib.Path(args.outputpath)
    else:
        base_dir = pathlib.Path.cwd() / 'logs' / 'local_runs'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if 'resnet' in args.arch and args.resnet_head:
            variant_stub = args.resnet_head
        elif args.alc2d and 'resnet' in args.arch:
            variant_stub = args.resnet_alc_variant
        elif args.chophead and 'resnet' in args.arch:
            variant_stub = 'chophead'
        else:
            variant_stub = 'base'
        run_stub = sanitize_name(args.run_name) or sanitize_name(f"{args.arch}_{variant_stub}")
        output_dir = base_dir / f"{run_stub}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def get_resnet_alc_model(base_model, args):
    variant_builders = {
        'full': ResNetWithALC_14x14,
        'channel-separate': ResNetWithALC_14x14_ChannelSep,
        'modified-stride': ResNetWithALC_ModifiedStride,
    }
    model_cls = variant_builders[args.resnet_alc_variant]
    print(f"=> Applying AdaptiveLocal2DLayer to ResNet with variant '{args.resnet_alc_variant}'")
    return model_cls(base_model, num_classes=1000), args.resnet_alc_variant


def get_resnet_head_model(base_model, args):
    head_builders = {
        'chophead': ResNetChopHead14x14,
        'coordconv': ResNetWithCoordConvHead14x14,
        'localconnected': ResNetWithLocalConnectedHead14x14,
        'deformconv': ResNetWithDeformConvHead14x14,
    }
    model_cls = head_builders[args.resnet_head]
    print(f"=> Applying ResNet head '{args.resnet_head}' at the 14x14 layer3 interface")
    return model_cls(base_model, num_classes=1000), args.resnet_head


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_run_metadata(args, model):
    return {
        'run_name': args.run_name or None,
        'arch': args.arch,
        'data': str(args.data),
        'pretrained': args.pretrained,
        'freeze_base': args.freeze_base,
        'alc2d': args.alc2d,
        'resnet_alc_variant': getattr(args, 'resolved_resnet_alc_variant', None),
        'resnet_head': args.resnet_head or None,
        'chophead': args.chophead,
        'scheduler': args.scheduler,
        'optimizer': args.opt,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'workers': args.workers,
        'outputpath': str(args.outputpath),
        'trainable_parameters': count_trainable_parameters(model),
        'total_parameters': sum(param.numel() for param in model.parameters()),
    }


def save_run_metadata(args, metadata):
    metadata_path = pathlib.Path(args.outputpath) / 'run_metadata.json'
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"=> Run metadata saved to {metadata_path}")


def save_run_summary(args, summary):
    summary_path = pathlib.Path(args.outputpath) / 'run_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"=> Run summary saved to {summary_path}")


def main():
    args = parser.parse_args()
    args.outputpath = resolve_output_dir(args)
    args.resolved_resnet_alc_variant = None
    print(f"=> Using output directory: {args.outputpath}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def chop_head(themodel,toremove=2):
    # Create an empty Sequential object
    # Define the number of layers to remove (e.g., remove the last 3 layers)
    num_layers_to_remove = toremove
    #print(themodel)
    # Get the internal module list
    modules = themodel._modules
    #print(modules[:-num_layers_to_remove])
    # Loop through and remove the unwanted modules
    for i in range(num_layers_to_remove):
        del modules[str(list(modules.keys())[-1])]
   
    # Update the model's internal module list
    themodel._modules = modules
    # Update model's attribute references
    for i, module_name in enumerate(modules.keys()):
        setattr(themodel, module_name, modules[module_name])
    
   
    print(themodel)
    return themodel


# def gauss_lr(epoch, lr, center, min_lr=1e-5, max_lr=1e-2, lr_sigma=0.25):
#     import numpy as np
#     print("testing", epoch, lr, min_lr, max_lr, center, lr_sigma)
#     lr = (min_lr + max_lr * np.exp(-(epoch-center)**2 / (center*lr_sigma)**2))
#     return lr

def ReplacePool(base_model, device='cpu', dynamic=False, args=None):
   # print(base_model)
    

    avg_pool = base_model._modules["avgpool"]
    
    channels = base_model.fc.in_features # feature dim is the channels coming from layer4
    # or get it using model_ft.layer4[1].conv2.out_channels for resnet
    print(device)
    #s = input()
    # Define the desired new average pooling layer
    if not dynamic:
        print("=> replacing the poollayer w Adaptive Pool")
        new_avg_pool = AdaptiveGaussianMasked2DPooling(output_size=(1,1), channels=channels, device=device)
    else:
        print("=> replacing the poollayer w Dynamic Pool")
        new_avg_pool = AdaptiveGaussianDynamic2DPooling(output_size=(1,1), channels=channels, device=device)

    # Replace the existing layer with the new one
    base_model._modules["avgpool"] = new_avg_pool
    
    # Make all layers trainable by default
    #for param in base_model.parameters():
    #    param.requires_grad = False
    
    # for param in base_model.fc.parameters():
    #     param.requires_grad = True
    
    # for param in base_model.avgpool.parameters():
    #     param.requires_grad = True
    
    
    #s=summary(base_model, input_size=(args.batch_size, 3, 224, 224),  
    #          col_names=["input_size","kernel_size", "output_size", "num_params", "mult_adds"])
    #print(s)
    return base_model


def ReplaceOutput(base_model, device='cpu', dynamic=False, args=None):
    class_layer = base_model._modules["fc"]
    
    in_features = class_layer.in_features
    out_features = class_layer.out_features
    # Define the desired new average pooling layer
    if not dynamic:
        print("=> replacing the classification layer")
        new_fc = AdaptiveLocal1D(in_features, out_features, device=device)
    else:
        print("=> replacing the poollayer w Dynamic Pool")
        new_fc = DynamicLocal1D(in_features, out_features,  device=device)

    # Replace the existing layer with the new one
    base_model._modules["avgpool"] = new_fc

    return base_model

# Define your custom model by removing the top layers and adding your own layers
class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes, removehead=2, device='cpu'):
        super(CustomModel, self).__init__()
        # Load the pre-trained model
        self.base_model = base_model #models.resnet50(pretrained=True)
        print("Removing:",list(self.base_model.children())[-removehead:])
        last_layer_shape = list(self.base_model.children())[-1].in_features
        #input("heyeyy: "+str(last_layer_shape)) 
        # Remove the top layers
        self.features = nn.Sequential(*list(self.base_model.children())[:-removehead])

        # Add your own layers
        #(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        #(fc): Linear(in_features=512, out_features=1000, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #device = self.features[-1].get_device()
        #self.avgpool = AdaptiveGaussianMasked2DPooling(output_size=(1,1)).to(device)
        self.fc = nn.Linear(in_features=last_layer_shape,out_features= num_classes)
        print("new model .................")
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        #model = models.__dict__[args.arch](pretrained=True)
        model = models.__dict__[args.arch](weights="DEFAULT")
        
        # =========================================================================
        # MODIFICATI<ON 1: FREEZE BASE MODEL LAYERS
        # If the 'freeze_base' flag is set, we freeze all parameters of the
        # pre-trained model before adding the new layers.
        # =========================================================================
        if args.freeze_base:
            print("=> Freezing all layers of the pre-trained base model...")
            for param in model.parameters():
                param.requires_grad = False
            print("=> Base model frozen.")
        
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    
    #print(model)
    
    def get_last_layer(model):
        if hasattr(model, 'fc'):
            return model.fc
        elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
            return model.heads.head
        elif hasattr(model, 'classifier'):
            return model.classifier
        else:
            raise AttributeError("Cannot find classification head in model.")
    
    if args.alc2d:
        print("=> Injecting AdaptiveLocal2DLayer...", args.alc2d)        

        if "resnet" in args.arch:
            model, args.resolved_resnet_alc_variant = get_resnet_alc_model(model, args)

        elif "vit" in args.arch:
            print("=> Applying AdaptiveLocal2DLayer to ViT")
            alc2d_output = (16,16)
            base_vit = model
            if  args.arch == "vit_b_16":
                
                patch_size = 16
            elif  args.arch == "vit_b_32":
                
                patch_size = 32
            elif  args.arch == "vit_l_16":
                
                patch_size = 16
            elif  args.arch == "vit_l_32":
                
                patch_size = 32
            else:
                raise ValueError(f"Unknown model: {args.arch}")
           
            model = ViTWithALC(base_vit,  num_classes=1000, patch_size=patch_size, 
                               alc_output= alc2d_output)
            

        else:
            raise ValueError(f"=> Unsupported architecture for --alc2d: {args.arch}")

    elif args.resnet_head and "resnet" in args.arch:
        model, args.resolved_resnet_alc_variant = get_resnet_head_model(model, args)

    elif args.chophead and "resnet" in args.arch:
        print("=> Using ResNet with layer4 removed and standard classifier")
        model = ResNetChopHead14x14(model, num_classes=1000)
        args.resolved_resnet_alc_variant = 'chophead-14x14'
        
    elif args.chophead and "vit" in args.arch:
        model = ViTStandardHead(model, num_classes=1000)
         

        # optionally add elif for vit...
    
    
    if args.gausspool:
        print("=> replacing the poollayer Adaptive Avg Gauss Pool")
        #model = chop_head(model)
        model = ReplacePool(model, device=args.gpu, dynamic=False, args=args)
    
    if args.dynpool:
        
        #model = chop_head(model)
        model = ReplacePool(model, device=args.gpu, dynamic=True, args=args)
    
    # if args.chophead:
    #     print("=> chopping the head")
    #     #model = chop_head(model)
    #     model = CustomModel(model, num_classes=1000, removehead=args.chophead,device=args.gpu)
        

    last_layer = get_last_layer(model)
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    
    #criterion = nn.CrossEntropyLoss().to(device)
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()  # Add this line for AMP
    
    if args.L1Softmax>0:
        class CustomLoss(torch.nn.Module):
            def __init__(self, criterion, l1_lambda, last_layer):
                super(CustomLoss, self).__init__()
                self.criterion = criterion
                self.l1_lambda = l1_lambda
                self.last_layer = last_layer

            def forward(self, outputs, targets):
                original_loss = self.criterion(outputs, targets)
                l1_loss = torch.norm(self.last_layer.weight, p=1)
                total_loss = original_loss + self.l1_lambda * l1_loss
                return total_loss

        # Use the custom loss function
        l1_lambda = args.L1Softmax  # Adjust this value based on your preference
        criterion = CustomLoss(criterion, l1_lambda, last_layer)

        # Modify the optimizer to only update the parameters of the last layer
        #optimizer = torch.optim.SGD([{'params': last_layer.parameters(), 'weight_decay': 0}],  # Set weight_decay to 0 for L1 on the last layer
        #                    lr=args.lr, momentum=args.momentum)

 
    print("=> Configuring optimizer...")
    params_to_train = []
    trainable_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_train.append(param)
            trainable_param_names.append(name)

    if args.log_optimizer_params:
        for name in trainable_param_names:
            print(f"\tAdding to optimizer: {name}")
    else:
        print(f"=> Trainable parameter tensors: {len(trainable_param_names)}")
    
    if not params_to_train:
        raise ValueError("No trainable parameters found. If you froze the base model, ensure new layers were added.")

    optimizer=None
    if (args.opt).lower() =='sgd':
        optimizer = torch.optim.SGD(params_to_train, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opt.lower() =='adamw' :
        args.weight_decay = 0.3
        optimizer = torch.optim.AdamW(params_to_train,
                                      lr=args.lr,                 
                                      weight_decay=args.weight_decay,
                                      eps=1e-8
                                    )
    else:
        print("unknown optimizer", args.opt)
            
    print("=> Optimizer", optimizer, " configured.")
        


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        S = 1000
        train_dataset = datasets.FakeData(1281167//S, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000//S, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.scheduler == 'gauss' and args.gaussLR:
        print("Applying Gauss LR starting from", args.lr, "Peaking at x", args.lr * args.gaussLR)
        scheduler = GaussianLR(optimizer, args.epochs // 3, min_lr=args.lr, max_lr=args.lr * args.gaussLR)
    
    elif args.scheduler == 'cosine':
        if args.batch_size > 256:
            # Calculate warmup epochs: max(5, 5% of total epochs)
            warmup_epochs = max(5, int(0.05 * args.epochs))
            print(f"Using CosineAnnealingLR with {warmup_epochs}-epoch warmup (batch_size={args.batch_size} > 256)")
            
            # Sequential scheduler: warmup -> cosine
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
                    CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
                ],
                milestones=[warmup_epochs]
            )
        else:
            print("Using CosineAnnealingWarmRestarts (default)")
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    
    else:
        print("Using StepLR")
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    #print(model)
    model_metadata = build_run_metadata(args, model)
    save_run_metadata(args, model_metadata)
    print(f"=> Trainable parameters: {model_metadata['trainable_parameters']:,}")
    if model_metadata['resnet_alc_variant']:
        print(f"=> ResNet variant: {model_metadata['resnet_alc_variant']}")
    try:
        s = summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "kernel_size", "output_size", "num_params"])
        if hasattr(s, 'total_params'):
            model_metadata['torchinfo_total_params'] = int(s.total_params)
            save_run_metadata(args, model_metadata)
    except Exception as exc:
        print(f"=> Skipping torchinfo summary: {exc}")
    #print(s)
    #input("Here!:")

    if args.evaluate:
        eval_summary = validate(val_loader, model, criterion, args)
        save_run_summary(args, {
            'mode': 'evaluate',
            'arch': args.arch,
            'run_name': args.run_name or None,
            'resnet_variant': args.resolved_resnet_alc_variant,
            'resnet_head': args.resnet_head or None,
            'pretrained': args.pretrained,
            'epochs_completed': 0,
            'best_acc1': float(eval_summary['top1']),
            'last_val': eval_summary,
        })
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        print("Model will train for epoch:", epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)

        current_lr = scheduler.get_last_lr()
        
        print(f'Learning Rate: {current_lr}')

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scaler, epoch, device, args)

        # evaluate on validation set
        val_summary = validate(val_loader, model, criterion, args)
        acc1 = val_summary['top1']
        
        scheduler.step()
        
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'run_name': args.run_name,
                'resnet_alc_variant': args.resolved_resnet_alc_variant,
                'outputpath': str(args.outputpath),
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, path=args.outputpath)

        save_run_summary(args, {
            'mode': 'train',
            'arch': args.arch,
            'run_name': args.run_name or None,
            'resnet_variant': args.resolved_resnet_alc_variant,
            'resnet_head': args.resnet_head or None,
            'pretrained': args.pretrained,
            'epochs_completed': epoch + 1,
            'best_acc1': float(best_acc1),
            'last_val': val_summary,
        })


def train(train_loader, model, criterion, optimizer, scaler, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    #print(model)
    
    
    if args.gausspool:
        def statshook():
            #print("Yes here I am ")
            #return
            import numpy as np
            gausspool_stats, _ = model.module.avgpool.stats([np.max, np.mean, np.min])
            return(gausspool_stats)
    else:
        def statshook():
            return

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.max_train_batches and i >= args.max_train_batches:
            print(f"=> Reached max_train_batches={args.max_train_batches}; stopping epoch early.")
            break
        
        # print some stats
        
        
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast():  # Enable mixed precision
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()

        # Apply gradient clipping (e.g., to all parameters)
        unscaled_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_value_(unscaled_params, clip_value=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Apply constraints (if relevant to AdaptiveLocal2DLayer)
        # Only apply if model has the layers
        for name, module in model.named_modules():
            if hasattr(module, "apply_constraints"):
                module.apply_constraints()

        
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
           progress.display(i + 1)
            #print(statshook())
            


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                if args.max_val_batches and i >= args.max_val_batches:
                    print(f"=> Reached max_val_batches={args.max_val_batches}; stopping validation early.")
                    break
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                with autocast():
                    output = model(images)
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
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return {
        'top1': float(top1.avg),
        'top5': float(top5.avg),
        'loss': float(losses.avg),
        'avg_batch_time': float(batch_time.avg),
    }


        
import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def save_checkpoint(state, is_best, filename=None, path=''):
    if filename is None:        
        filename = f"checkpoint_{state['arch']}_{timestamp}_latest.pth.tar"


    full_path = os.path.join(path, filename)
    print(f"=> Saving checkpoint to {full_path}")
    torch.save(state, full_path)

    if is_best:
        best_path = os.path.join(path, f"model_best_{state['arch']}.pth.tar")
        shutil.copyfile(full_path, best_path)
        print(f"=> New best model saved to {best_path}")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
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
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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
    s = 0
    print("Sleeping: ",s," secs")
    time.sleep(s)
    main() 
    
    #pretrained
    # python inet_trainer_alc2d.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --pretrained --freeze_base --opt adamW --lr 1e-3  --scheduler cosine --epoch 5 --b 192 --alc2d
 
    #python add training from scratch calls. 
    ## Use cosine annealing
    
    # python inet_trainer_alc2d.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --pretrained --scheduler cosine --epochs 90 --lr 0.01 --b 256

    
    # python inet_trainer_alc2d.py --dummy  -a resnet50 --pretrained --lr 1e-3  --epoch 5 --b 192 --alc2d True
    #python inet_trainer_alc2d.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --lr 1e-4 --epoch 30 --b 128  --alc2d True
    # python inet_trainer_alc2d.py --dummy  -a vit_b_32 --pretrained --lr 1e-3  --epoch 5 --b 192 --alc2d True
    #python inet_trainer.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --lr 1e-4 --gaussLR 30 --epoch 30 --b 192 --gausspool 1 >> gauss_pool_from_scratch_9_04_2024_gausslr_30.txt
    # python inet_trainer.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --pretrained --lr 1e-4 --epoch 5 --b 192 --gausspool 1 >> lastrun_gp2.txt
    # just evaluate
    # just dummy
    #python inet_trainer.py --dummy  -a resnet50 --pretrained --lr 1e-3  --epoch 5 --b 192 --gausspool 1
    # --resume checkpoint.pth.tar
    # python inet_trainer.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --pretrained --lr 1e-4 --epoch 0 --evaluate --b 192 --dynpool True >> lastrun_gp3.txt
    # python inet_trainer.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --pretrained --lr 1e-4 --epoch 0 --evaluate --b 192 --dynpool True >> lastrun_gp3.txt
    # python inet_trainer.py "C:\datasets\ILSVRC\Data\CLS-LOC" -a resnet50 --lr 1e-4 --gaussLR 20 --epoch 20 --b 192 --gausspool True >> gausspool_run_from_scrath_8_4_2024.txt
    
