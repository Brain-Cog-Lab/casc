import torch
import torch.nn as nn
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000000
# matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse, logging, datetime, time

from src.utils import seed_all, accuracy, AverageMeter, result2csv, setup_default_logging, mergeConvBN
from src.clipquantization import replace_relu_by_cqrelu
from src.convertor import *
from src.dataset import *
from src.model import *


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--save_result', type=bool, default=True)
parser.add_argument('--device', type=str, default='1')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='cifar100', choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--model', type=str, default='vgg16', help="'cifarconvnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50'")

'''
/data/ly/casc/20221027-093907-vgg16-cifar10-max-True-8-42
/data/ly/casc/20221027-073056-vgg16-cifar100-max-True-8-42
/data/ly/casc/20221031-225011-resnet20-cifar10-avg-True-8-42
/data/ly/casc/20221031-224954-resnet20-cifar100-avg-True-8-42

/data/ly/casc/20221027-075937-vgg16-cifar10-max-True-16-42
/data/ly/casc/20221027-061952-vgg16-cifar100-max-True-16-42
/data/ly/casc/20221031-221332-resnet20-cifar100-avg-True-16-42
/data/ly/casc/20221031-221351-resnet20-cifar10-avg-True-16-42

/data0/ly/casc/20230205-035558-resnet34-imagenet-avg-True-8-42
/data0/ly/casc/20230217-181932-vgg16-imagenet-avg-True-8-42
'''

parser.add_argument('--ckpt_path', type=str, default="//data/ly/casc/20221027-073056-vgg16-cifar100-max-True-8-42")
parser.add_argument('--neg', default=True, type=bool, help='negtive spikes')
# parser.add_argument('--neg', action='store_true', help='negtive spikes')
parser.add_argument('--sleep', default=8, type=int, help='sleep time')
parser.add_argument('--margin', default=8, type=float, help='sleep margin')
parser.add_argument('--T', default=8, type=int, help='simulation time')

parser.add_argument('--cqrelu', type=bool, default=True)
# parser.add_argument('--cqrelu', action='store_true')
parser.add_argument('--qlevel', type=int, default=32)
parser.add_argument('--pool', type=str, default='max')
parser.add_argument('--data_path', type=str, default='/data/datasets', help='/Users/lee/data/datasets')
parser.add_argument('--saved_dir', type=str, default='/data/ly/casc/conversion/')
parser.add_argument('--saved_csv', type=str, default='./result_cls_conversion.csv')
parser.add_argument('--train_batch', default=30, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for testing')

parser.add_argument('--seed', default=14, type=int, help='seed')

parser.add_argument('--p', default=0.999, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=1, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--lipool', default=True, type=bool, help='LIPooling')
parser.add_argument('--soft_mode', default=True, type=bool, help='soft_reset')
parser.add_argument('--channelnorm', default=False, type=bool, help='channelnorm')
# parser.add_argument('--pseudo_convert', default=False, type=bool, help='pseudo_convert')
parser.add_argument('--pseudo_convert', action='store_true')
parser.add_argument('--merge', default=True, type=bool, help='fuseConvBN')
args = parser.parse_args()


def evaluate_net(data_iter, net, device):
    net = net.to(device)
    acc_tot = AverageMeter()
    with torch.no_grad():
        start = time.time()
        for ind, data in enumerate(data_iter):
            print("\r", end='')
            print("eval %d/%d ...:" % (ind, len(data_iter)), end='')
            X = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            net.eval()
            output = net(X.to(device)).detach()
            net.train()

            acc, = accuracy(output, y.to(device), topk=(1,))
            acc_tot.update(acc, output.shape[0])

    print('<Test>   acc:%.6f, time:%.1f s' % (acc_tot.avg, time.time() - start))
    return acc_tot.avg.detach().cpu().item()

def close_bias(model):
    children = list(model.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, (nn.Conv2d, nn.Linear)) and hasattr(child, 'bias'):
            child.bias.data *= 0
        else:
            close_bias(child)


def debug_evaluate_snn(data_iter, net, netno, ann, device, T, args):
    net = net.to(device)
    netno = netno.to(device)

    acc_tot = AverageMeter()
    with torch.no_grad():
        dict1 = deepcopy(net).state_dict()
        net2 = deepcopy(net)
        close_bias(net2)
        dict2 = net2.state_dict()

        start = time.time()
        cnt, error = 0,0
        for ind, data in enumerate(data_iter):
            # print("eval %d/%d ...:" % (ind, len(data)), end='\r')
            X = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            net.eval()
            output, acc_t = 0, []
            outputno = 0
            net.reset()
            netno.reset()
            for t in range(T):
                # output += net.features[:10](deepcopy(X).to(device)).detach()
                output += net(deepcopy(X).to(device)).detach()
                outputno += netno(deepcopy(X).to(device)).detach()
                # acc, = accuracy(output / (t+1), y.to(device), topk=(1,))
                # acc_t += [acc.detach().cpu().item()]

                if (t+1) % args.margin == 0 and args.sleep != 0:
                # if t+1==T:
                    net.change_sleep()
                    net.load_state_dict(dict2)
                    for ti in range(args.sleep):
                        # output += net.features[:10](torch.zeros_like(X).to(device)).detach()
                        output += net(torch.zeros_like(X).to(device)).detach()
                        # acc, = accuracy(output / (t + 1), y.to(device), topk=(1,))
                        # acc_t += [acc.detach().cpu().item()]
                    net.load_state_dict(dict1)
                    net.change_sleep()

            # 2 5 9 12 16 19 22 26 29 32 36 39 42
            index = 42
            # out_snn = net.features[index][0].sumspike / T
            # # out_mem = net.features[index][0].summem / T
            # out_ann = ann.features[:index+1](deepcopy(X).to(device))
            snet = net.features[index][0]
            anet = ann.features[:index+1]

            aa = snet.sumspike / T
            bb = anet(deepcopy(X).to(device))

            a = ((bb - aa)**2).view(args.batch_size, -1).mean(1).detach().cpu().numpy()
            b = (bb**2).view(args.batch_size, -1).mean(1).detach().cpu().numpy()

            # print("ralerror:" )
            for i in range(args.batch_size):
                print(a[i]/b[i])
            if ind >=9:
                break

            # out_snn = netno.conv1[5][0].sumspike / T
            # out_ann = ann.conv1[:6](deepcopy(X).to(device))
    #
    #         # error += ((out_snn - out_ann).abs() > 0.01).sum()
    #         error += (out_snn < out_ann).sum()
    #
    #         # error_less = ((out_mem-out_snn)>=1/T).sum()
    #         # error_more = (((out_mem-out_snn)<0) & (out_mem>=0)).sum()
    #         # error += error_more + error_less
    #         cnt += out_ann.numel()
    #
    # print(cnt, error, error/cnt *100)



if __name__ == '__main__':
    print(args.ckpt_path)
    args.model = args.ckpt_path.split('-')[-6]
    args.dataset = args.ckpt_path.split('-')[-5]
    args.pool = 'avg'
    # args.pool = args.ckpt_path.split('-')[-4]
    args.cqrelu = args.ckpt_path.split('-')[-3] == 'True'
    args.qlevel = int(args.ckpt_path.split('-')[-2])
    # args.margin = args.qlevel

    logger = logging.getLogger('train')
    seed_all(seed=args.seed)
    device = torch.device("cuda:%s" % args.device) if args.use_cuda else 'cpu'

    now = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
    args.time = now
    exp_name = '-'.join([
        now,
        args.dataset,
        args.model,
        str(args.cqrelu),
        str(args.qlevel),
        ])
    if args.save_result: os.makedirs(args.saved_dir, exist_ok=True)
    setup_default_logging(log_path=os.path.join(args.saved_dir, exp_name+'.txt') if args.save_result else None)

    args_dict = "Namespace:\n"
    for eachArg, value in args.__dict__.items():
        args_dict += eachArg + ' : ' + str(value) + '\n'
    logger.info(args_dict)

    train_loader = eval('get_%s_data' % args.dataset)(
        args.data_path, args.train_batch, train=True, num_workers=8)
    test_loader = eval('get_%s_data' % args.dataset)(
        args.data_path, args.batch_size, train=False, num_workers=8)
    in_channels = 3

    num_classes = cls_num_classes[args.dataset]

    if args.model == 'vgg16':
        model = VGG16(pool=args.pool)
        if args.dataset == 'imagenet':
            model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            model.fc = nn.Linear(25088, num_classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet20':
        model = ResNet20(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet34':
        model = ResNet34(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError

    if args.cqrelu:
        model = replace_relu_by_cqrelu(model, args.qlevel)

    if args.dataset != 'imagenet':
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'checkpoints.pth'), map_location=device))
        # model.load_state_dict({k: v for k, v in torch.load(os.path.join(args.ckpt_path, 'checkpoints.pth')).items()})
    else:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.ckpt_path, 'checkpoints.pth')).items()})

    model = mergeConvBN(model) if args.merge else model
    model = model.to(device)
    # acc = evaluate_net(test_loader, model, device)
    # args.acc_ann = acc

    if args.cqrelu:
        converter = CQConvertor(
            soft_mode=args.soft_mode,
            lipool=args.lipool,
            gamma=args.gamma,
            pseudo_convert=args.pseudo_convert,
            merge=args.merge,
            neg=args.neg,
            sleep_time=[args.qlevel, args.qlevel+args.sleep])

        converterno = CQConvertor(
            soft_mode=args.soft_mode,
            lipool=args.lipool,
            gamma=args.gamma,
            pseudo_convert=args.pseudo_convert,
            merge=args.merge,
            neg=False,
            sleep_time=[args.qlevel, args.qlevel])
    else:
        converter = PercentConvertor(
            dataloader=train_loader,
            device=device,
            p=args.p,
            channelnorm=args.channelnorm,
            soft_mode=args.soft_mode,
            lipool=args.lipool,
            gamma=args.gamma,
            pseudo_convert=args.pseudo_convert,
            merge=args.merge,
            neg=args.neg)

    snn = converter(deepcopy(model))
    snn_no = converterno(deepcopy(model))

    debug_evaluate_snn(test_loader, snn, snn_no, model, device, args.T, args)
    # args_dict = vars(args)
    # args_dict.update({"acc_best": max(acc)})
    # args_dict.update({"best_ind": np.argmax(acc)})
    # for i in range(len(acc)):
    #     args_dict.update({"acc_%d"%(i+1) : acc[i]})


