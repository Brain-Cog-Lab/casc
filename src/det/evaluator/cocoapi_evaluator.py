import json
import tempfile

import torch
from ..data.coco import *
from ..data.transforms import ValTransforms
import os
from ..utils.misc import CollateIndFunc, CollateFunc
from ..utils.spiking_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time
try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


def close_bias(model):
    children = list(model.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, (nn.Conv2d, nn.Linear)) and hasattr(child, 'bias'):
            child.bias.data *= 0
        else:
            close_bias(child)


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, img_size, device, testset=False, transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.testset = testset
        if self.testset:
            image_set = 'test2017'
        else:
            image_set = 'val2017'

        self.dataset = COCODataset(
                            data_dir=data_dir,
                            image_set=image_set,
                            img_size=img_size,
                            transform=ValTransforms(img_size))
        self.img_size = img_size
        self.transform = transform
        self.device = device

        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.

    def evaluate(self, model, batch_size=16, T=256, smode=False):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
            batch_size: bs for inference
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))
        # start testing

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateIndFunc(),
            num_workers=8,
            pin_memory=True
        )

        start = time.time()
        for iter_i, (images, _, h_all, w_all, scale_all, offset_all, indx) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            with torch.no_grad():
                if smode:
                    clean_mem_spike(model)
                    model.obj_p, model.cls_p, model.reg_p, model.t = 0, 0, 0, 0
                    for _ in range(T):
                        bbox_all, score_all, cls_ind_all = model.spiking_forward(images)
                else:
                    bbox_all, score_all, cls_ind_all = model(images)
            for i in range(images.shape[0]):
                bboxes = bbox_all[i]
                scores = score_all[i]
                cls_inds = cls_ind_all[i]
                h = h_all[i]
                w = w_all[i]
                id_ = int(indx[i])
                ids.append(id_)

                # map the boxes to original image
                bboxes -= offset_all[i]
                bboxes /= scale_all[i]
                size = np.array([[w, h, w, h]])
                bboxes *= size

                for i, box in enumerate(bboxes):
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    label = self.dataset.class_ids[int(cls_inds[i])]

                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    score = float(scores[i])  # object score * class score
                    A = {"image_id": id_, "category_id": label, "bbox": bbox,
                         "score": score}  # COCO json format
                    data_dict.append(A)
        annType = ['segm', 'bbox', 'keypoints']

        end = time.time()
        print("time:", end-start)

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('coco_test-dev.json', 'w'))
                cocoDt = cocoGt.loadRes('coco_test-dev.json')
                return -1, -1
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
                cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
                cocoEval.params.imgIds = ids
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
                print('ap50_95 : ', ap50_95)
                print('ap50 : ', ap50)
                self.map = ap50_95
                self.ap50_95 = ap50_95
                self.ap50 = ap50

                return ap50, ap50_95
        else:
            return 0, 0

    def evaluate_T(self, model, batch_size=16, T=256, args=None):
        model.eval()

        dict1 = deepcopy(model).state_dict()
        net2 = deepcopy(model)
        close_bias(net2)
        dict2 = net2.state_dict()

        self.ids = [[] for _ in range(int(T+(T//args.margin)*args.sleep))]
        self.data_dict = [[] for _ in range(int(T+(T//args.margin)*args.sleep))]
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))
        # start testing

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateIndFunc(),
            num_workers=8,
            pin_memory=True
        )

        start = time.time()
        for iter_i, (images, _, h_all, w_all, scale_all, offset_all, indx) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            bboxs, scores, clsinds = [], [], []
            with torch.no_grad():  
                model.reset()
                model.obj_p, model.cls_p, model.reg_p, model.t = 0, 0, 0, 0
                for tt in range(T):
                    model.t = tt + 1
                    bbox_all, score_all, cls_ind_all = model.spiking_forward(images)
                    bboxs.append(bbox_all)
                    scores.append(score_all)
                    clsinds.append(cls_ind_all)

                    if (tt + 1) % args.margin == 0 and args.sleep != 0:
                        model.change_sleep()
                        model.load_state_dict(dict2)
                        for ti in range(args.sleep):
                            bbox_all, score_all, cls_ind_all = model.spiking_forward(torch.zeros_like(images).to(self.device))
                            bboxs.append(bbox_all)
                            scores.append(score_all)
                            clsinds.append(cls_ind_all)
                        model.change_sleep()
                        model.load_state_dict(dict1)

            for iii in range(len(self.ids)):
                self.ids[iii], self.data_dict[iii] = self.get_parse(images.shape[0], bboxs[iii], scores[iii], clsinds[iii], h_all, w_all, indx, scale_all, offset_all, self.ids[iii], self.data_dict[iii])
            # if iter_i >= 3:
            #     break

        annType = ['segm', 'bbox', 'keypoints']
        end = time.time()
        print("time:", end-start)

        a = []
        # file_name = args.backbone + '_' + args.dataset + '_' + str(args.channelnorm) + '_' + str(args.gamma) + '_' + str(args.p) + '_' + str(args.spicalib) + '_' + str(args.allowance)
        # f = open('./result/new/' + file_name + '.txt', 'w')
        for timestep in range(len(self.ids)):
            if timestep+1 in [2,4,8,16,32,64,128]:
                print(timestep+1)
                # Evaluate the Dt (detection) json comparing with the ground truth
                if len(self.data_dict[timestep]) > 0:
                    print('evaluating ......')
                    cocoGt = self.dataset.coco
                    # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
                    _, tmp = tempfile.mkstemp()
                    json.dump(self.data_dict[timestep], open(tmp, 'w'))
                    cocoDt = cocoGt.loadRes(tmp)
                    cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
                    cocoEval.params.imgIds = self.ids[timestep]
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()

                    ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
                    print('TimeStep:', timestep)
                    print('ap50_95 : ', ap50_95)
                    print('ap50 : ', ap50)
                    self.map = ap50_95
                    self.ap50_95 = ap50_95
                    self.ap50 = ap50

                    # f.write(str(self.ap50) + ',')
                    # f.flush()
                    a += [self.map]
                    # return ap50, ap50_95
                else:
                    a += [0.0]
        # f.close()
        return a


    def evaluate_T_debug(self, model, ann, batch_size=16, T=256, args=None):
        model.eval()

        dict1 = deepcopy(model).state_dict()
        net2 = deepcopy(model)
        close_bias(net2)
        dict2 = net2.state_dict()

        self.ids = [[] for _ in range(int(T+(T//args.margin)*args.sleep))]
        self.data_dict = [[] for _ in range(int(T+(T//args.margin)*args.sleep))]
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))
        # start testing

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateIndFunc(),
            num_workers=8,
            pin_memory=True
        )

        start = time.time()
        for iter_i, (images, _, h_all, w_all, scale_all, offset_all, indx) in enumerate(dataloader):
            images = images.to(self.device)
            bboxs, scores, clsinds = [], [], []
            with torch.no_grad():
                model.reset()
                model.obj_p, model.cls_p, model.reg_p, model.t = 0, 0, 0, 0
                for tt in range(T):
                    model.t = tt + 1
                    bbox_all, score_all, cls_ind_all = model.spiking_forward(images)

                    if (tt + 1) % args.margin == 0 and args.sleep != 0:
                        model.change_sleep()
                        model.load_state_dict(dict2)
                        for ti in range(args.sleep):
                            bbox_all, score_all, cls_ind_all = model.spiking_forward(torch.zeros_like(images).to(self.device))

                        model.change_sleep()
                        model.load_state_dict(dict1)

            # 关于VGGbackbone的
            # 2 5 9 12 16 19 22 26 29 32 36 39 42
            # index = 2
            # snet = model.backbone[index][0]
            # aa = snet.sumspike / T
            #
            # anet = ann.backbone[:index + 1]
            # bb = anet(deepcopy(images))

            # 关于后续结构neck
            # aa = model.neck[1].convs[2][0].sumspike / T
            # bb = ann.neck(ann.backbone(images))

            # 关于后续结构reg_head, 前面0-4，中间2/5
            aa = model.reg_feat[3].convs[2][0].sumspike / T
            tmp = ann.neck(ann.backbone(images))
            bb = ann.reg_feat[:4](tmp)

            a = ((bb - aa) ** 2).view(args.batch_size, -1).mean(1).detach().cpu().numpy()
            b = (bb ** 2).view(args.batch_size, -1).mean(1).detach().cpu().numpy()

            for i in range(args.batch_size):
                print(a[i] / b[i])
                torch.cuda.empty_cache()
            if iter_i >= 199:
                break



    def get_parse(self,shape0, bbox_all, score_all, cls_ind_all, h_all, w_all, indx, scale_all, offset_all, ids, data_dict):
        for i in range(shape0):
            bboxes = bbox_all[i]
            scores = score_all[i]
            cls_inds = cls_ind_all[i]
            h = h_all[i]
            w = w_all[i]
            id_ = int(indx[i])
            ids.append(id_)

            # map the boxes to original image
            bboxes -= offset_all[i]
            bboxes /= scale_all[i]
            size = np.array([[w, h, w, h]])
            bboxes *= size

            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(cls_inds[i])]

                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i])  # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                        "score": score}  # COCO json format
                data_dict.append(A)
        return ids, data_dict

if __name__ == '__main__':
    dataset = COCODataset(
        data_dir='/home/hexiang/COCO/',
        image_set='train2017',
        img_size=640,
        transform=None)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=CollateFunc(),
        num_workers=8,
        pin_memory=True
    )

    for iter_i, (images, targets) in enumerate(dataloader):
        h, w, _ = images.shape