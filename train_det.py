from __future__ import division

import os
import random
import argparse
import time
import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import logging
from src.det.data.voc import VOCDetection
from src.det.data.coco import COCODataset
from src.det.data.transforms import TrainTransforms, ColorTransforms, ValTransforms
from tqdm import tqdm


from src.det.utils import distributed_utils
from src.det.utils.com_flops_params import FLOPs_and_Params
from src.det.utils.misc import ModelEMA
from src.det.utils.misc import CollateFunc
from src.det.utils import create_labels
from src.det.utils.criterion import build_criterion

from src.det.evaluator.cocoapi_evaluator import COCOAPIEvaluator
from src.det.evaluator.vocapi_evaluator import VOCAPIEvaluator

from src.utils import result2csv, seed_all, setup_default_logging
from src.clipquantization import replace_relu_by_cqrelu


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')

    # basic
    parser.add_argument('--device', default='4', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--max_epoch', type=int, default=200,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[120, 170], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                        default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', default=False, type=bool,
                        help='use tensorboard')
    parser.add_argument('--debug', default=False, type=bool,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='./', type=str,
                        help='Gamma update for SGD')
    parser.add_argument('--backbone', default='VGG16', type=str,
                        help='backbone used for ann', choices=['VGG16', 'ResNet50'])

    # input image size
    parser.add_argument('--img_size', type=int, default=640,
                        help='The size of input image')

    # Loss
    parser.add_argument('--loss_obj_weight', default=1.0, type=float,
                        help='weight of obj loss')
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')
    parser.add_argument('--scale_loss', default='batch', type=str,
                        help='scale loss: batch or positive samples')

    # parser.add_argument('--cqrelu', action='store_true')
    parser.add_argument('--cqrelu', type=bool, default=True)
    parser.add_argument('--qlevel', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--saved_dir', type=str, default='/data/ly/casc_det')
    parser.add_argument('--saved_csv', type=str, default='/data/ly/casc_det/results_det.csv')
    # parser.add_argument('--save_log', type=bool, default=False)
    parser.add_argument('--save_log', action='store_false')

    # train trick
    parser.add_argument('-ms', '--multi_scale', default=True, type=bool,
                        help='use multi-scale trick')
    parser.add_argument('-no_wp', '--no_warm_up', default=False, type=bool,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--ema', default=True, type=bool,
                        help='use ema training trick')
    parser.add_argument('--mosaic', default=False, type=bool,
                        help='use Mosaic Augmentation trick')
    parser.add_argument('--mixup', default=True, type=bool,
                        help='use MixUp Augmentation trick')
    parser.add_argument('--multi_anchor', default=False, type=bool,
                        help='use multiple anchor boxes as the positive samples')
    parser.add_argument('--center_sample', default=False, type=bool,
                        help='use center sample for labels')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='accumulate gradient')

    # DDP train
    parser.add_argument('-dist', '--distributed', default=False, type=bool,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', default=False, type=bool,
                        help='use sybn.')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    return parser.parse_args()


def train():
    args = parse_args()
    seed_all(args.seed)
    args.logger = logging.getLogger('train')

    now = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
    args.time = now
    exp_name = '-'.join([
        now,
        "yolo",
        args.backbone,
        args.dataset,
        str(args.cqrelu),
        str(args.qlevel),
        str(args.seed)])

    args.saved_dir = os.path.join(args.saved_dir, exp_name)
    if args.save_log: os.makedirs(args.saved_dir, exist_ok=True)
    setup_default_logging(log_path=os.path.join(args.saved_dir, 'log.txt') if args.save_log else None)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["OMP_NUM_THREADS"] = "1"  # 设置OpenMP计算库的线程数
    os.environ["MKL_NUM_THREADS"] = "1"  # 设置MKL-DNN CPU加速库的线程数。
    torch.set_num_threads(1)

    if args.distributed:
        distributed_utils.init_distributed_mode(args)

    args_dict = "Namespace:\n"
    for eachArg, value in args.__dict__.items():
        args_dict += eachArg + ' : ' + str(value) + '\n'
    args.logger.info(args_dict)
    args.logger.info("----------------------------------------------------------")
    # path_to_save = os.path.join(args.save_folder, args.backbone + '_' + args.dataset)
    # os.makedirs(path_to_save, exist_ok=True)

    # 是否使用cuda
    if args.cuda:
        args.logger.info('use cuda: '+ str(args.device))
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 是否使用多尺度训练
    if args.multi_scale:
        args.logger.info('use the multi-scale trick ...')
    train_size = val_size = args.img_size

    # 构建dataset类和dataloader类
    if args.dataset == 'voc':
        # 加载voc数据集
        data_dir = '/data/datasets/VOC_2007/VOCdevkit/'
        num_classes = 20
        dataset = VOCDetection(
                        data_dir=data_dir,
                        img_size=train_size,
                        transform=TrainTransforms(train_size),
                        color_augment=ColorTransforms(train_size),
                        mosaic=args.mosaic,
                        mixup=args.mixup)

        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size))

    elif args.dataset == 'coco':
        # 加载COCO数据集
        data_dir = '/data/datasets/COCO/'
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size,
                    image_set='train2017',
                    transform=TrainTransforms(train_size),
                    color_augment=ColorTransforms(train_size),
                    mosaic=args.mosaic,
                    mixup=args.mixup)

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size)
                        )

    else:
        args.logger.info('unknow dataset !! Only support voc and coco !!')
        exit(0)

    if args.distributed:
        # 2.使用DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=(train_sampler is None), collate_fn=CollateFunc(),
                                                 sampler=train_sampler,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True
                                                 )
    else:
        # dataloader类
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=CollateFunc(),
            num_workers=args.num_workers,
            pin_memory=True
        )

    args.logger.info('Training model on: %s' % args.dataset)
    args.logger.info('The dataset size: %d' % len(dataset))
    args.logger.info("----------------------------------------------------------")

    from src.det.models.yolo import YOLO
    yolo_net = YOLO(device, img_size=train_size, num_classes=num_classes, trainable=True, center_sample=args.center_sample, backbone=args.backbone)
    args.logger.info('Let us train yolo on the %s dataset ......' % (args.dataset))
    # args.logger.info("our yolo")
    args.logger.info(yolo_net)

    model = yolo_net
    model = model.to(device).train()
    model_without_ddp = model


    if args.distributed:
        if args.sybn:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        args.logger.info('using DDP....')
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_without_ddp.trainable = False
        model_without_ddp.eval()
        FLOPs_and_Params(model=model_without_ddp,
                         img_size=train_size,
                         device=device)
        model_without_ddp.trainable = True
        model_without_ddp.train()

    # # keep training
    # if args.resume is not None:
    #     args.logger.info('keep training model: %s' % (args.resume))
    #     model.load_state_dict(torch.load(args.resume, map_location=device))
    if args.dataset=='voc' and args.backbone=="ResNet50":
        args.resume = "/data/ly/casc/20221027-205350-yolo-ResNet50-voc-False-128-42/checkpoints.pth"
        model.load_state_dict(torch.load(args.resume, map_location=device))
    elif args.dataset=='coco' and args.backbone=="ResNet50":
        args.resume = "/data/ly/casc/20221120-101435-yolo-ResNet50-coco-False-128-42/checkpoints.pth"
        model.load_state_dict(torch.load(args.resume, map_location=device))


    if args.cqrelu:
        model = replace_relu_by_cqrelu(model, args.qlevel)


    # 使用 tensorboard 可视化训练过程
    if args.tfboard:
        args.logger.info('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    # EMA
    ema = ModelEMA(model_without_ddp) if args.ema else None

    # optimizer setup
    base_lr = args.lr
    tmp_lr = args.lr

    # optimizer
    optimizer = optim.SGD(model_without_ddp.parameters(),
                          lr=tmp_lr,
                          momentum=0.9,
                          weight_decay=5e-4)

    epoch_size = len(dataset) // (args.batch_size * torch.cuda.device_count())  # 每一训练轮次的迭代次数
    best_map = -100.
    best_map50 = -100.
    warmup = not args.no_warm_up

    # build_criterion
    cfg = {}  # 简单设为{}
    criterion = build_criterion(args, cfg, num_classes)

    # 开始训练
    t0 = time.time()
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch in args.lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + (epoch + 1) * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                args.logger.info('Warmup is over !!')
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # 多尺度训练
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # 随机选择一个新的尺寸
                train_size = random.randint(10, 20) * 32
                model_without_ddp.set_grid(train_size)
            if args.multi_scale:
                # 插值
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

            # 制作训练标签
            targets = [label.tolist() for label in targets]
            # make labels
            targets = create_labels.gt_creator(
                                    img_size=train_size,
                                    strides=model_without_ddp.stride,
                                    label_lists=targets,
                                    anchor_size=None,
                                    multi_anchor=False,
                                    center_sample=args.center_sample)
            # to device
            images = images.to(device)
            targets = targets.to(device)

            # inference
            pred_obj, pred_cls, pred_iou, targets = model_without_ddp(images, targets=targets)

            # compute loss
            loss_obj, loss_cls, loss_reg, total_loss = criterion(pred_obj, pred_cls, pred_iou, targets)

            # check loss
            if torch.isnan(total_loss):
                continue

            loss_dict = dict(
                loss_obj=loss_obj,
                loss_cls=loss_cls,
                loss_reg=loss_reg,
                total_loss=total_loss
            )

            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
            #args.logger.info("iter %d: reduced not stuck" % iter_i)
            total_loss = total_loss / args.accumulate
            # Backward and Optimize
            total_loss.backward()
            #args.logger.info("iter %d: backward not stuck" % iter_i)
            optimizer.step()
            optimizer.zero_grad()
            #args.logger.info("iter %d: optim step not stuck" % iter_i)
            # ema
            if args.ema:
                ema.update(model_without_ddp)
            #args.logger.info("iter %d: ema updata not stuck" % iter_i)
            if distributed_utils.is_main_process() and iter_i % 20 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', loss_dict_reduced['loss_obj'].item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', loss_dict_reduced['loss_cls'].item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', loss_dict_reduced['loss_reg'].item(), iter_i + epoch * epoch_size)
                t1 = time.time()
                args.logger.info(
                    "[Epoch %d/%d][Iter %d/%d][lr %.6f][Loss: obj %.2f || cls %.2f || reg %.2f || size %d || time: %.2f]" % (
                    epoch + 1, args.max_epoch, iter_i, epoch_size, tmp_lr,
                    loss_dict_reduced['loss_obj'].item(),
                    loss_dict_reduced['loss_cls'].item(),
                    loss_dict_reduced['loss_reg'].item(),
                    train_size,
                    t1 - t0))

                t0 = time.time()


        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == args.max_epoch:
            if evaluator is None:
                args.logger.info('No evaluator ... save model and go on training.')
                args.logger.info('Saving state, epoch: {}'.format(epoch + 1))
                weight_name = 'epoch_{}.pth'.format(epoch + 1)
                torch.save(model_without_ddp.state_dict(), os.path.join(args.saved_dir, weight_name))
            else:
                args.logger.info('eval ...')
                # check ema
                if args.ema:
                    model_eval = ema.ema
                else:
                    model_eval = model.module if args.distributed else model

                # set eval mode
                model_eval.trainable = False
                model_eval.set_grid(val_size)
                model_eval.eval()

                # check evaluator
                if distributed_utils.is_main_process():
                        # evaluate
                        evaluator.evaluate(model_eval)

                        cur_map = evaluator.map
                        if cur_map > best_map:
                            # update best-map
                            best_map = cur_map
                            best_map50 = evaluator.ap50

                            # save model
                            args.logger.info('Saving state, epoch: %d, map: %.2f' % (epoch + 1, best_map*100))
                            # weight_name = 'epoch_{}_{:.2f}.pth'.format(epoch + 1, best_map * 100)
                            torch.save(model_eval.state_dict(), os.path.join(args.saved_dir, 'checkpoints.pth'))

                if args.distributed:
                    # wait for all processes to synchronize
                    dist.barrier()

                    # set train mode.
                model_eval.trainable = True
                model_eval.set_grid(train_size)
                model_eval.train()

        # close mosaic augmentation
        if args.mosaic and args.max_epoch - epoch == 15:
            args.logger.info('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False
        # close mixup augmentation
        if args.mixup and args.max_epoch - epoch == 15:
            args.logger.info('close Mixup Augmentation ...')
            dataloader.dataset.mixup = False

    args.ap50_95 = best_map * 100
    args.ap50 = best_map50 * 100
    result2csv(args.saved_csv, args)
    print("Write results to csv file.")

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()