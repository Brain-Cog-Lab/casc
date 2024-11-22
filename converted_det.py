import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse, logging, datetime, time, os

from src.utils import seed_all, accuracy, AverageMeter, result2csv, setup_default_logging, mergeConvBN
from src.clipquantization import replace_relu_by_cqrelu
from src.convertor import *
from src.dataset import *
from src.model import *

from src.det.data.voc import VOCDetection
from src.det.data.coco import COCODataset
from src.det.data.transforms import TrainTransforms, ColorTransforms, ValTransforms
from src.det.evaluator.cocoapi_evaluator import COCOAPIEvaluator
from src.det.evaluator.vocapi_evaluator import VOCAPIEvaluator
from src.det.utils.com_flops_params import FLOPs_and_Params


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--save_result', type=bool, default=True)
parser.add_argument('--device', type=str, default='3')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'coco'])
parser.add_argument('--model', type=str, default='VGG16', choices=['ResNet50', 'VGG16'])

'''
/data/ly/casc/20221102-100752-yolo-VGG16-voc-True-4-42
/data/ly/casc/20221102-100832-yolo-VGG16-voc-True-8-42/
/data/ly/casc/20221102-100917-yolo-VGG16-voc-True-16-42
/data/ly/casc/20221102-101016-yolo-VGG16-voc-True-32-42

/data/ly/casc/20221031-234132-yolo-ResNet50-voc-True-4-42
/data/ly/casc/20221031-092006-yolo-ResNet50-voc-True-8-42
/data/ly/casc/20221030-110639-yolo-ResNet50-voc-True-16-42
/data0/ly/casc/20221029-181623-yolo-ResNet50-voc-True-32-42

/data/ly/casc/20221114-135924-yolo-VGG16-coco-True-4-42
/data/ly/casc/20221114-135924-yolo-VGG16-coco-True-8-42
/data/ly/casc/20221114-135924-yolo-VGG16-coco-True-16-42


'''

parser.add_argument('--ckpt_path', type=str, default="/data/ly/casc/20221114-135924-yolo-VGG16-coco-True-4-42")
parser.add_argument('--neg', default=True, type=bool, help='negtive spikes')
# parser.add_argument('--neg', action='store_true', help='negtive spikes')
parser.add_argument('--sleep', default=4, type=int, help='sleep time')
parser.add_argument('--margin', default=4, type=float, help='sleep margin')
parser.add_argument('--T', default=4, type=int, help='simulation time')

parser.add_argument('--cqrelu', type=bool, default=True)
# parser.add_argument('--cqrelu', action='store_true')
parser.add_argument('--qlevel', type=int, default=4)
parser.add_argument('--data_path', type=str, default='/data/datasets', help='/Users/lee/data/datasets')
parser.add_argument('--saved_dir', type=str, default='/data/ly/casc/conversion/')
parser.add_argument('--saved_csv', type=str, default='./result_det_conversion.csv')
parser.add_argument('--train_batch', default=30, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=8, type=int, help='batch size for testing')

parser.add_argument('--img_size', type=int, default=640, help='The size of input image')

parser.add_argument('--seed', default=14, type=int, help='seed')
parser.add_argument('--p', default=0.999, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=1, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--lipool', default=True, type=bool, help='LIPooling')
parser.add_argument('--soft_mode', default=True, type=bool, help='soft_reset')
parser.add_argument('--channelnorm', default=False, type=bool, help='channelnorm')
parser.add_argument('--pseudo_convert', default=False, type=bool, help='pseudo_convert')
parser.add_argument('--merge', default=True, type=bool, help='fuseConvBN')
args = parser.parse_args()


if __name__ == '__main__':
    args.backbone = args.ckpt_path.split('-')[-5]
    args.dataset = args.ckpt_path.split('-')[-4]
    args.cqrelu = args.ckpt_path.split('-')[-3] == 'True'
    args.qlevel = int(args.ckpt_path.split('-')[-2])

    seed_all(seed=args.seed)
    device = torch.device("cuda:%s" % args.device) if args.use_cuda else 'cpu'

    train_size = val_size = args.img_size

    if args.dataset == 'voc':
        # 加载voc数据集
        data_dir = '/data/datasets/VOC_2007/VOCdevkit/'
        num_classes = 20

        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size))

    elif args.dataset == 'coco':
        # 加载COCO数据集
        data_dir = '/data/datasets/COCO/'
        num_classes = 80

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size)
                        )

    else:
        args.logger.info('unknow dataset !! Only support voc and coco !!')
        exit(0)

    from src.det.models.yolo import YOLO
    yolo_net = YOLO(device, img_size=train_size, num_classes=num_classes, trainable=True, center_sample=False, backbone=args.backbone)

    model = yolo_net
    model = model.to(device)

    if args.cqrelu:
        model = replace_relu_by_cqrelu(model, args.qlevel)

    model.trainable = False
    model.eval()
    FLOPs_and_Params(model=model,
                     img_size=train_size,
                     device=device)

    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'checkpoints.pth'), map_location=device))
    model = mergeConvBN(model) if args.merge else model
    model_eval = model

    model_eval.trainable = False
    model_eval.set_grid(val_size)
    model_eval.eval()

    evaluator.evaluate(model_eval, args.batch_size)
    args.map = evaluator.map
    print("map: %.6f"% args.map)

    if args.cqrelu:
        converter = CQConvertor(
            soft_mode=args.soft_mode,
            lipool=args.lipool,
            gamma=args.gamma,
            pseudo_convert=args.pseudo_convert,
            merge=args.merge,
            neg=args.neg,
            sleep_time=[args.qlevel, args.qlevel+args.sleep])

    snn = converter(deepcopy(model))
    maps = evaluator.evaluate_T(snn, args.batch_size, args.T, args)

    args_dict = vars(args)
    args_dict.update({"map_best": max(maps)})
    args_dict.update({"best_ind": np.argmax(maps)})
    for i in range(len(maps)):
        args_dict.update({"acc_%d" % i: maps[i]})

    if args.save_result:
        result2csv(args.saved_csv, args_dict)
