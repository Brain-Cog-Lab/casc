import sys

import torchvision.models

sys.path.append("../..")
import matplotlib
matplotlib.use('agg')

from torch.nn.parallel import DistributedDataParallel as DDP

from src.dataset import *
from src.neuron import *
from src.clipquantization import replace_relu_by_cqrelu
from src.model import *
from src.utils import result2csv, seed_all, setup_default_logging, accuracy, AverageMeter, EMA

import matplotlib.pyplot as plt
import datetime, time, argparse, logging, math, os

try:
    from apex import amp
except:
    print('no apex pakage installed')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--model', type=str, default='vgg16',
                        help="'cifarconvnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50', 'regnet_x_8gf', 'resnext50_32x4d', 'mobilenet'")
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--pool', type=str, default='avg')
    parser.add_argument('--data_path', type=str, default='/data/datasets', help='/Users/lee/data/datasets')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--cqrelu', action='store_true')
    # parser.add_argument('--cqrelu', default=True, type=bool)
    parser.add_argument('--qlevel', type=int, default=8)

    # Optimizer and lr_scheduler
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--optim', type=str, default='sgd', choices=['adamW', 'adam', 'sgd'])
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse'])
    parser.add_argument('--schedu', type=str, default='cosin', choices=['step', 'mstep', 'cosin'])
    parser.add_argument('--step_size', type=int, default=50, help='parameter for StepLR')
    parser.add_argument('--milestones', type=list, default=[150, 250])
    parser.add_argument('--lr_gamma', type=float, default=0.05)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--warmup_lr_init', type=float, default=1e-6)
    parser.add_argument('--init_lr', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--ema', type=float, default=None)

    # Path
    parser.add_argument('--resume', type=str,
                        default='')
    # parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--saved_dir', type=str, default='/data/ly/casc')
    parser.add_argument('--saved_csv', type=str, default='./results.csv')
    parser.add_argument('--save_log', type=bool, default=False)
    # parser.add_argument('--save_log', action='store_false')

    # Device
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--apex', type=bool, default=False)
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dali', type=bool, default=False)
    parser.add_argument('--channel_last', type=bool, default=False)

    return parser.parse_args()


def train_net(net, train_iter, test_iter, optimizer, scheduler, criterion, device, args=None):
    print("Start training...")
    best = 0
    net = net.to(device)
    class_num = args.num_classes

    if args.ema is not None:
        ema = EMA(model, 0.999)
        ema.register()
        args.ema = ema

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        loss_tot = AverageMeter()
        acc_tot = AverageMeter()
        start = time.time()
        net.train()

        for ind, data in enumerate(train_iter):
            # if ind >= len(train_iter)//100:
            #     break
            if args.local_rank == int(args.device):
                tim = int(time.time()-start)
                pert = tim/(ind+1)
                eta = pert * (len(train_iter)-ind)
                print("\r", end='')
                print("Training iter:", str(ind)+'/'+str(len(train_iter)),
                      '['+ '%02d:%02d' % (tim//60, tim%60)+'<'+
                      '%02d:%02d,' % (eta // 60, eta % 60),
                      '%.2f s/it]' % pert,
                      end=''
                      )

            X = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            output = net(X)
            label = F.one_hot(y, class_num).float() if isinstance(criterion, torch.nn.MSELoss) else y
            loss = criterion(output, label)

            optimizer.zero_grad()
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

            elif args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()
                if args.ema is not None:
                    args.ema.update()

            acc, = accuracy(output, y, topk=(1,))
            loss_tot.update(loss.item(), output.shape[0])
            acc_tot.update(acc, output.shape[0])

            if ind % 100 == 0:
                if args.local_rank == int(args.device):
                    args.logger.info('Epoch:%d, iter:%d, acc:%.6f, loss:%.6f, lr:%.6f, time:%.1f s'
                          % (epoch + 1, ind, acc_tot.avg, loss_tot.avg, optimizer.param_groups[0]['lr'], time.time() - start))

        scheduler.step()
        if args.local_rank == int(args.device):
            args.logger.info('-'*10+ 'Epoch:' + str(epoch + 1)+ '-'*10 + '\n' +
                '<Train>  acc:%.6f, loss:%.6f, lr:%.6f, time:%.1f s'
                  % (acc_tot.avg, loss_tot.avg, optimizer.param_groups[0]['lr'], time.time() - start))

        if args.saved_dir is not None:
            saved_dir = os.path.join(args.saved_dir, 'checkpoints.pth')

        if epoch % args.eval_every == 0:
            # test_acc, test_loss = evaluate_net(train_iter, net, criterion, device, args)
            test_acc, test_loss = evaluate_net(test_iter, net, criterion, device, args)

            if test_acc > best:
                best = test_acc

                if args.save_log and args.local_rank == int(args.device):
                    if args.ema is not None:
                        dict = {
                            "ema": args.ema,
                            "state_dict": net.state_dict()
                        }
                        torch.save(dict, saved_dir)
                    else:
                        torch.save(net.state_dict(), saved_dir)
                    args.logger.info("saved model on"+ saved_dir+"-"+ str(best))


        if args.local_rank == int(args.device):
            args.logger.info("<Best> acc:%.6f \n" % best)

        if args.save_log and not args.distributed:
            dic = {"epoch": epoch,
                   "train_loss": loss_tot.avg,
                   "test_loss": test_loss,
                   "train_acc" : acc_tot.avg.item(),
                   "test_acc": test_acc.item(),
                   }
            result2csv(os.path.join(args.saved_dir, 'result.csv'), dic)


    args.logger.info("Best test acc: %.6f" % best)

    args.acc = best.detach().cpu().item()
    result2csv(args.saved_csv, args)
    print("Write results to csv file.")


def evaluate_net(data_iter, net, criterion, device, args=None):
    class_num = args.num_classes if args is not None else 1000
    net = net.to(device)
    loss_tot = AverageMeter()
    acc_tot = AverageMeter()
    with torch.no_grad():
        # if args is not None and args.ema is not None: args.ema.apply_shadow()
        start = time.time()
        for ind, data in enumerate(data_iter):
            if type(args.local_rank) == list:
                args.local_rank = args.local_rank[0]
                args.device = args.device[0]

            if args is None:
                print("Testing", str(ind) + '/' + str(len(data_iter)), end='\r')
            elif args.local_rank == int(args.device):
                print("Testing", str(ind) + '/' + str(len(data_iter)), end='\r')

            X = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            net.eval()
            output = net(X.to(device)).detach()
            net.train()
            label = F.one_hot(y, class_num).float() if isinstance(criterion, torch.nn.MSELoss) else y
            loss = criterion(output, label)

            acc, = accuracy(output, y.to(device), topk=(1,))
            acc_tot.update(acc, output.shape[0])
            loss_tot.update(loss.item(), output.shape[0])

    if args is None:
        print('<Test>   acc:%.6f, time:%.1f s' % (acc_tot.avg, time.time() - start))
    elif args.local_rank == int(args.device):
        args.logger.info('<Test>   acc:%.6f, loss:%.6f, time:%.1f s' % (acc_tot.avg, loss_tot.avg, time.time() - start))
        if args.ema is not None: args.ema.restore()
    return acc_tot.avg, loss_tot.avg


if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)
    args.logger = logging.getLogger('train')

    now = (datetime.datetime.now()+datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
    args.time = now
    exp_name = '-'.join([
        now,
        args.model,
        args.dataset,
        args.pool,
        str(args.cqrelu),
        str(args.qlevel),
        str(args.seed)])

    args.saved_dir = os.path.join(args.saved_dir, exp_name)
    if args.save_log: os.makedirs(args.saved_dir, exist_ok=True)
    setup_default_logging(log_path=os.path.join(args.saved_dir, 'log.txt') if args.save_log else None)

    CKPT_DIR = os.path.join(args.saved_dir, exp_name)

    world_size = 1
    if not args.distributed:
        os.environ["LOCAL_RANK"] = args.device

    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank

    if args.local_rank == int(args.device):
        args_dict = "Namespace:\n"
        for eachArg, value in args.__dict__.items():
            args_dict += eachArg + ' : ' + str(value) + '\n'
        args.logger.info(args_dict)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)

        if args.distributed:
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend='nccl')
    else:
        device = torch.device("cpu")

    if args.dataset in ['mnist', 'fashionmnist']:
        train_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=True, num_workers=args.num_workers)
        test_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=False, num_workers=args.num_workers)
        in_channels = 1

    elif args.dataset in ['cifar10', 'cifar100', 'imagenet']:
        train_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=True,
            num_workers=args.num_workers, distributed=args.distributed)
        test_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=False,
            num_workers=args.num_workers, distributed=args.distributed)
        in_channels = 3
    elif args.dataset in ['dvsgesture', 'dvscifar10', 'ncaltech101', 'ncars', 'nmnist']:
        train_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, T=args.T, train=True, num_workers=args.num_workers)
        test_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, T=args.T, train=False, num_workers=args.num_workers)
        in_channels = 2
    else:
        raise NotImplementedError("Can't find the dataset loader")

    args.num_classes = num_classes = cls_num_classes[args.dataset]

    if args.model == 'vgg16':
        model = VGG16(pool=args.pool)
        if args.dataset == 'imagenet':
            model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            model.fc = nn.Linear(25088, num_classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet18':
        model = ResNet18(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet20':
        model = ResNet20(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet34':
        model = ResNet34(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet50':
        model = ResNet50(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet101':
        model = ResNet101(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet152':
        model = ResNet152(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'mobilenet':
        model = MobileNet()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet18v2':
        model = ResNet18v2()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet20v2':
        model = ResNet20v2()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=False)
        model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'regnet_x_8gf':
        model = torchvision.models.regnet_x_8gf(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError

    if args.resume != '':
        # args.logger.info('keep training model: %s' % args.resume)
        # model.load_state_dict(torch.load(args.resume, map_location=device))
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.resume, map_location=device).items()})

    if args.cqrelu:
        model = replace_relu_by_cqrelu(model, args.qlevel)

    if args.local_rank == int(args.device): args.logger.info(model)

    # args.init_lr = args.init_lr * args.batch_size * world_size / 1024.0
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.distributed:
        model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError

    if args.schedu == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.lr_gamma)
    elif args.schedu == 'mstep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.lr_gamma)
    elif args.schedu == 'cosin':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)
        warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup * (1 - args.warmup_lr_init) + args.warmup_lr_init if epoch < args.warmup \
            else args.min_lr + 0.5 * (1.0 - args.min_lr) * (math.cos((epoch - args.warmup) / (args.epochs - args.warmup) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    else:
        raise NotImplementedError

    print("Start training...")
    train_net(model,
              train_loader, test_loader,
              optimizer, scheduler, criterion,
              device,
              args)

    # test_acc, test_loss = evaluate_net(test_loader, model, criterion, device, args)
    # print(test_acc)