# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:#2种使用train.py文件的方法
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
    #通过cfg这个参数来传入你所要使用的网络结构，根据配置文件从0开始搭建一个，从头开始训练

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""
#导包操作
import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

#额外定义了3个变量：在后面训练时会频繁看到，尤其是RANK变量，变量的含义是做分部式训练用的，初学者一般是一台电脑一台gpu卡训练，不会涉及到分部式训练
#这些变量通常是默认的值

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# GIT_INFO = check_git_info()
#直接跳转到最下部分 if name main

#首先train函数会读取opt文件中的参数赋值给一些临时变量，供后续使用
def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir #定义了几个路径，save目录就是训练结果保存的目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir #判断weights文件夹是否存在，若不存在就创建一个weights文件夹
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters #加载和保存一些参数信息
    if isinstance(hyp, str): #加载训练过程中使用的超参数
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict #根据超参数的配置文件以键值对的形式加载进来
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items())) #把超参数打印出来
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings #把运行中的配置环境保存下来
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp) #把使用的超参数保存下来
        yaml_save(save_dir / 'opt.yaml', vars(opt)) #执行脚本的时候传入的参数和默认使用的参数都保存下来，opt这个文件

    # Loggers #定义了一个日志记录工具 w&b 和 Tensorboard 这两个库，训练过程的可视化操作，百度用法
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance #Ctrl+Loggers 进到_init_.py

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k)) #遍历日志记录器_init_.py中所有方法的时候，会将字符串和方法进行绑定
            #绑定的具体意义是什么callbacks.run('on_pretrain_routine_start') 根据字符串去日志记录器查找相应的函数，如果有就执行，没有就不执行
            #为了在训练过程中，在每个训练阶段控制训练日志的记录过程
        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots #plots赋值为true or false,控制训练过程中的图表画出来或训练结果画出来
    cuda = device.type != 'cpu' #判断电脑是否支持cuda
    init_seeds(opt.seed + 1 + RANK, deterministic=True) #初始随机化种子，便于每次random函数返回相同的值，保证每一次的训练过程是可复现的
    with torch_distributed_zero_first(LOCAL_RANK):# 与分部式训练相关的，若不进行，不执行此代码
        data_dict = data_dict or check_dataset(data)  # check if None #检查数据集是否存在，若coco128.yalm文件路径不存在，就会按照网址下载并解压到相应路径下，供训练使用
    train_path, val_path = data_dict['train'], data_dict['val'] #从coco128.yalm文件中取出训练集和验证集的路径赋值给train_path, val_path这两个变量
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes #从数据集中取出80个类名
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset #false

    # Model #模型加载相关
    check_suffix(weights, '.pt')  # check weights #检测传进来的weights的后缀名是否以.pt结尾
    pretrained = weights.endswith('.pt')
    if pretrained:#如果使用预训练权重的话，则执行以下代码
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally #先检测这个文件有没有，若没有会在yolov5官方仓库中下载yolov5s.pt
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak #预训练权重加载进来
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create #新建一个模型，nc是不同的
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32 #预训练模型的所有参数都加载进来csd
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect #判断csd与自己加载的模型有多少的参数是相同的
        model.load_state_dict(csd, strict=False)  # load #把相同的参数加载进来
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze #可以手动控制去冻结哪些层，默认为0，不冻结
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride) #获取模型最高层的特征，相较于原始输入图像，缩小了多少倍，gs：32
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # #检查输入图片的尺寸是否满足32的倍数，不满足的会自动补成32的倍数，来作为模型的输入图片大小

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        # #判断传入的batch_size大小是否为-1，自动帮助计算合适的batch_size大小，一般不会用到
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({'batch_size': batch_size})

    ###Optimizer #创建深度学习优化器
    #随机梯度下降算法SGD
    #对所有卷积层的w参数进行优化（需要进行权重衰减）
    nbs = 64  # nominal batch size #名义上的batch_size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing #专门求出accumulate变量用来存放累计次数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay #'weight_decay'这个权重衰减的超参数去进行缩放
    #权重衰减可以防止训练过程出现的过拟合，在超参数的配置文件中weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler #模型训练过程中学习率变化的策略
    #之前定义优化器的时候有传入初始学习率'lr0',但是根据深度学习的理论，学习率在训练过程中会随着训练的轮数逐渐降低，这样会更有利于找到局部最优解
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf'] #one_cycle的变化策略，余弦函数的变化周期
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear #线性的变化策略 lf学习率因子
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA #对模型使用指数移动平均，能在每次更新参数时考虑历史值对参数的影响，给训练过程带来帮助
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume #从预训练的权重文件中加载一些信息，除了保存每一层的网络结构参数外，也会有当时训练的其他信息
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd
    # 当时训练best_fitness拟合程度（评估模型训练好坏的指标），优化器信息等加载进来

    # DP mode #多张gpu训练时才会用到的，一般单卡训练是不会用到这些
    if cuda and RANK == -1 and torch.cuda.device_count() > 1: #判断是否用了多张显卡
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model) #数据并行化的操作

    # SyncBatchNorm #与分部式训练相关
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader #加载训练集数据
    #自定义训练集的数据集和所使用的数据加载器，数据加载涉及的内容比较多，科科考虑在后面加节课讲解
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class #计算标签的最大类别号，例coco128 0-79，即mlc：79
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:  #加载验证集数据
        # 自定义验证集的数据集和所使用的数据加载器
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor #自动检查anchors，可以自动调整anchors大小
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names) #日志记录的功能

    # DDP mode #多卡训练的功能
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps) nl：3
    # #从模型中取出检测层的数量，网络结构中的低层，中层，高层3个层，利用这个层数对超参数进行缩放
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    #hyp['box']，hyp['cls']，hyp['obj']这几个超参数，损失函数前面的因子，系数.    来把这3个超参数缩放到和3个层级时一样的尺度
    hyp['label_smoothing'] = opt.label_smoothing #做标签平滑时的超参数，默认不用
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    #把nc：类别数量 hyp:超参数 class_weights:类别权重 names：标签名 写入到model对应的变量中
    # Start training #开始整个的训练过程
    #先进行初始化
    t0 = time.time()  #t0：用来统计训练一轮需要的时间
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training #nw：warmup的迭代次数
    last_opt_step = -1 # last_opt_step：上一次更新参数时计数器的值，即批次号，优化器涉及
    maps = np.zeros(nc)  # mAP per class #用来存放训练过程中计算出的每一类的map值 80个maps
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move #必须了解pytorch源码才能清楚，初学者可以略过
    scaler = torch.cuda.amp.GradScaler(enabled=amp) #训练过程中使用自动混合精度去训练
    stopper, stop = EarlyStopping(patience=opt.patience), False #EarlyStopping连续训练几轮，模型的效果都没有得到提升的话，提前终止
    compute_loss = ComputeLoss(model)  # init loss class #定义了损失函数
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')  #打印提示信息
    for epoch in range(start_epoch, epochs):  # epoch #for循环遍历整个300轮的训练过程------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only) #更新模型的权重
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            #数据集中每一类的数量权重，如果每一类的数量比较多的话，其权重会比较大，增加其被采样到的概率
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            #把类别权重换算到图像的维度，每一张图像的采样权重，比如某一张图片含有识别不精确的目标数量越多，这个图片的权重越大。
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            #利用图片权重进行随机的重采样，不是原来数量的数据集了，多包含一些难识别的样本

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses #初始化mloss变量来存放损失值
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar #训练过程的进度条
        optimizer.zero_grad() #梯度归零
        for i, (imgs, targets, paths, _) in pbar:  # batch #遍历每一个batch，一批批取数据------------------------------------------------------------
            #imgs图像数据，eg.batch_size=16,imgs=16取出16张图像数据，后续传给模型进行预测的
            # targets,标注框，标注框和预测框一起求损失函数值 path：图片的路径，方便做可视化的工作
            callbacks.run('on_train_batch_start') #每一批数据开始训练的时候会记录一些信息
            ni = i + nb * epoch  # number integrated batches (since train start)
            #ni表示从第0轮开始到目前为止，总共训练了多少批数据，起到记录批次的功能
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0#把图片移到gpu上进行归一化操作

            # Warmup
            # #warmup也是一种训练技巧，刚开始训练前几批数据的时候，用一个小的学习率，慢慢升到初始学习率
            if ni <= nw: #当前的批次正好处于需要进行warmup的前几批数据时
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):#遍历优化器中所有的参数组
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale #多尺度训练，需要指定才会用到
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                # #训练过程中随机化得到一个比例因子，用这个因子去改变训练过程中输入图片的尺寸，从而起到多尺度训练的效果
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward #前向传播
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward #把图片传给模型，得到预测框pred
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                #利用预测框和标注框计算损失值
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward #反向传播
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html #更新参数
            if ni - last_opt_step >= accumulate: #对多批数据累累积统一做一次更新
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log #终端更新进度条信息，日志记录
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))#前三批会保存训练集上的结果train_batch0，1，2,每一批的效果图
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler #这一轮的所有批次训练完成后，会根据之前定义的学习率变化策略，更新一下学习率
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP #当1个epoch结束后，会在验证集上计算一个mAP及其他指标
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            #给ema添加这几个属性
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            #判断当前这一轮是否为最终的一轮，如果不是最终的一轮，会把目前训练好的这一轮的模型在验证集上跑一下
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)
                #result: 7个值，maps：80个类别各自的map值

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            #定义了fitness拟合度指标，衡量模型目前训练的好坏程度，对results中的7个指标做了加权求和。
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:#判断当前的拟合度是否为最好的拟合度
                best_fitness = fi #记录下当前最好的拟合度
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
            #日志记录，把这一轮的结果保存下来，写入到results.csv文件中

            # Save model #并判断是否把模型保存下来，保存模型
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': 'GIT_INFO',  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last) #先把本轮的训练结果保存为last.pt
                if best_fitness == fi: #如果本轮的拟合度就是最好的拟合度，就把best.pt也保存成这一轮的模型
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training  #当所有300轮训练结束后-----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}: #当所有300轮训练结束后，将训练结果最好的权重单独取出来，在验证集上跑一遍，并将最终结果打印出来
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')#完成所有训练花费了多少时间
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best: #best.pt 在验证集上跑一遍
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache() #显存释放
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'pretrained/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT /'models/yolov5s_he3.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/person.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=600, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()): #main函数分成4个部分：
    # Checks #第1部分：校验工作
    if RANK in {-1, 0}: #不进行分部式训练的话RANK默认值为-1，会执行后续的3行代码
        print_args(vars(opt)) #打印文件所用到的参数信息，参数包括命令行传入的参数，及默认的一些参数
        check_git_status() #检查yolov5 github仓库中的代码是否更新，更新的话在这里会提示
        check_requirements() #检查requirements.txt中python依赖包是否安装成功，如果没有安装成功的话也会给予提示

    # Resume (from specified or most recent last.pt) #第2部分：根据命令行中是否传入resume参数，来执行不同的操作，resume从中断中恢复
    if opt.resume and not check_comet_resume(opt) and not opt.evolve: #首先会判断在命令行中是否传入resume这个参数，由于并没有传入resume参数，会执行else
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else: #执行else
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        #检查几个配置文件的路径
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified' #判断cfg和weights是否都为空，都为空会报错
        if opt.evolve: #判断是否传入了evolve这个参数，如果传入，训练保存目录会更改为'runs/evolve'
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg': #判断是否传入name参数，默认exp，增量路径保存
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode #第3部分：判断是否采用DDP这种训练方法
    device = select_device(opt.device, batch_size=opt.batch_size) #选择cpu还是gpu
    if LOCAL_RANK != -1: #假如是分部式训练会采用额外操作，一般不会用到分部式训练
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    # Train #第4部分：开始正式进行训练
    if not opt.evolve: #先判断是否使用evolve参数，未使用evolve参数，调用train函数执行模型训练过程
        train(opt.hyp, opt, device, callbacks)  #####重点只有train 函数
#遗传算法（突变）———进化超参数的方法，耗时耗费计算资源，大多是人就是使用默认的超参数+手动调参
    # Evolve hyperparameters (optional)
    else: #传入evolve参数，则执行else，一般人不会用到
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv),])

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__': #执行了2个函数
    opt = parse_opt() #第1个函数：解析训练过程中用到的参数
    main(opt) #第2个函数：执行main函数，main函数分成4个部分

