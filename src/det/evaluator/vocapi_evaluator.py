"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from ..data.voc import VOCDetection, VOC_CLASSES
import os
import time
import numpy as np
import pickle
import xml.etree.ElementTree as ET
from ..utils.misc import CollateIndFunc
from torch.utils.data import DataLoader
from tqdm import tqdm, trange  
import torch
import torch.nn as nn
from copy import deepcopy
from ..utils import AverageMeter


def close_bias(model):
    children = list(model.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, (nn.Conv2d, nn.Linear)) and hasattr(child, 'bias'):
            child.bias.data *= 0
        else:
            close_bias(child)


class VOCAPIEvaluator():
    """ VOC AP Evaluation class """
    def __init__(self, 
                 data_dir, 
                 img_size, 
                 device, 
                 transform, 
                 set_type='test', 
                 year='2007', 
                 display=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.device = device
        self.transform = transform
        self.labelmap = VOC_CLASSES
        self.set_type = set_type
        self.year = year
        self.display = display

        # path
        self.devkit_path = os.path.join(data_dir, 'VOC' + year)
        self.annopath = os.path.join(data_dir, 'VOC2007', 'Annotations', '%s.xml')
        self.imgpath = os.path.join(data_dir, 'VOC2007', 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(data_dir, 'VOC2007', 'ImageSets', 'Main', set_type+'.txt')
        self.output_dir = self.get_output_dir('voc_eval/', self.set_type)

        # dataset
        self.dataset = VOCDetection(data_dir=data_dir, 
                                    image_sets=[('2007', set_type)],
                                    transform=transform)

    def evaluate(self, net, batch_size=16, T=256, smode=False):
        net.eval()
        num_images = len(self.dataset)
        self.all_boxes = [[[] for _ in range(num_images)]
                        for _ in range(len(self.labelmap))]

        # timers
        det_file = os.path.join(self.output_dir, 'detections.pkl')

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateIndFunc(),
            num_workers=8,
            pin_memory=True
        )

        start = time.time()
        for iter_i, (images, _, h_all, w_all, scale_all, offset_all, _) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            with torch.no_grad():
                if False:
                    pass
                else:
                    bbox_all, score_all, cls_ind_all = net(images)
             
            for i in range(images.shape[0]):
                bboxes = bbox_all[i]
                scores = score_all[i]
                cls_inds = cls_ind_all[i]
                h = h_all[i]
                w = w_all[i]

                # map the boxes to original image
                bboxes -= offset_all[i]
                bboxes /= scale_all[i]
                size = np.array([[w, h, w, h]])
                bboxes *= size

                for j in range(len(self.labelmap)):
                    inds = np.where(cls_inds == j)[0]
                    if len(inds) == 0:
                        self.all_boxes[j][iter_i*batch_size+i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = bboxes[inds]
                    c_scores = scores[inds]
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(np.float32,
                                                                         copy=False)
                    self.all_boxes[j][iter_i*batch_size+i] = c_dets
            # if iter_i>20:
            #     break

        end = time.time()
        print("time:", end-start)
        # with open(det_file, 'wb') as f:
        #     pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self.evaluate_detections(self.all_boxes)

        print('Mean AP: ', self.map)


    def evaluate_T(self, net, batch_size=16, T=32, args=None):
        net.eval()

        dict1 = deepcopy(net).state_dict()
        net2 = deepcopy(net)
        close_bias(net2)
        dict2 = net2.state_dict()

        num_images = len(self.dataset)
        # self.alls = [[[[] for _ in range(num_images)]
        #                 for _ in range(len(self.labelmap))]
        #                 for _ in range(1)]

        self.alls = [[[[] for _ in range(num_images)]
                        for _ in range(len(self.labelmap))]
                        for _ in range(int(T+(T//args.margin)*args.sleep))]

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateIndFunc(),
            num_workers=8,
            pin_memory=True
        )

        start = time.time()
        for iter_i, (images, _, h_all, w_all, scale_all, offset_all, _) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            bboxs, scores, clsinds = [], [], []
            cou_t = 0
            with torch.no_grad():
                    net.reset()
                    net.obj_p, net.cls_p, net.reg_p, net.t = 0, 0, 0, 0
                    for tt in range(T):
                        net.t = tt + 1

                        bbox_all, score_all, cls_ind_all = net.spiking_forward(images)
                        bboxs.append(bbox_all)
                        scores.append(score_all)
                        clsinds.append(cls_ind_all)
                        cou_t += 1

                        if (tt + 1) % args.margin == 0 and args.sleep != 0:
                            net.change_sleep()
                            net.load_state_dict(dict2)
                            for ti in range(args.sleep):
                                bbox_all, score_all, cls_ind_all = net.spiking_forward(torch.zeros_like(images).to(self.device))
                                bboxs.append(bbox_all)
                                scores.append(score_all)
                                clsinds.append(cls_ind_all)
                                cou_t += 1
                            net.change_sleep()
                            net.load_state_dict(dict1)
            
            for iii in range(len(self.alls)):
                self.alls[iii] = self.get_parse(images.shape[0], bboxs[iii], scores[iii], clsinds[iii], h_all, w_all, scale_all, offset_all, iter_i, batch_size, self.alls[iii])

        end = time.time()
        print("time:", end-start)

        print('Evaluating detections')
        a = []
        for timestep in range(len(self.alls)):
            if timestep+1 in [2,4,8,16,32,64,128]:
                print(timestep+1)
                self.evaluate_detections(self.alls[timestep])
                a += [self.map]
        return a

    def evaluate_T_debug(self, net, ann, batch_size=16, T=32, args=None):
        net.eval()

        dict1 = deepcopy(net).state_dict()
        net2 = deepcopy(net)
        close_bias(net2)
        dict2 = net2.state_dict()

        num_images = len(self.dataset)
        # self.alls = [[[[] for _ in range(num_images)]
        #                 for _ in range(len(self.labelmap))]
        #                 for _ in range(1)]

        self.alls = [[[[] for _ in range(num_images)]
                      for _ in range(len(self.labelmap))]
                     for _ in range(int(T + (T // args.margin) * args.sleep))]

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateIndFunc(),
            num_workers=8,
            pin_memory=True
        )

        start = time.time()
        for iter_i, (images, _, h_all, w_all, scale_all, offset_all, _) in enumerate(dataloader):
            images = images.to(self.device)
            bboxs, scores, clsinds = [], [], []
            cou_t = 0
            with torch.no_grad():
                net.reset()
                net.obj_p, net.cls_p, net.reg_p, net.t = 0, 0, 0, 0
                for tt in range(T):
                    net.t = tt + 1
                    net.spiking_forward(images)
                    cou_t += 1

                    if (tt + 1) % args.margin == 0 and args.sleep != 0:
                        net.change_sleep()
                        net.load_state_dict(dict2)
                        for ti in range(args.sleep):
                            net.spiking_forward(torch.zeros_like(images).to(self.device))
                            cou_t += 1
                        net.change_sleep()
                        net.load_state_dict(dict1)

            # 关于VGGbackbone的
            # 2 5 9 12 16 19 22 26 29 32 36 39 42
            index = 42
            snet = net.backbone[index][0]
            aa = snet.sumspike / T

            anet = ann.backbone[:index + 1]
            bb = anet(deepcopy(images))

            # 关于后续结构neck
            # aa = net.neck[1].convs[2][0].sumspike / T
            # bb = ann.neck(ann.backbone(images))

            # 关于后续结构reg_head, 前面0-4，中间2/5
            # aa = net.reg_feat[0].convs[2][0].sumspike / T
            # tmp = ann.neck(ann.backbone(images))
            # bb = ann.reg_feat[:1](tmp)

            a = ((bb - aa) ** 2).view(args.batch_size, -1).mean(1).detach().cpu().numpy()
            b = (bb ** 2).view(args.batch_size, -1).mean(1).detach().cpu().numpy()

            for i in range(args.batch_size):
                print(a[i]/b[i])
                torch.cuda.empty_cache()
            if iter_i >=199:
                break



    def get_parse(self, shape0, bbox_all, score_all, cls_ind_all, h_all, w_all, scale_all, offset_all, iter_i, batch_size, all_boxes):
        for i in range(shape0):
            bboxes = bbox_all[i]
            scores = score_all[i]
            cls_inds = cls_ind_all[i]
            h = h_all[i]
            w = w_all[i]

            bboxes -= offset_all[i]
            bboxes /= scale_all[i]
            size = np.array([[w, h, w, h]])
            bboxes *= size

            for j in range(len(self.labelmap)):
                inds = np.where(cls_inds == j)[0]
                if len(inds) == 0:
                    all_boxes[j][iter_i*batch_size+i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes,
                                    c_scores[:, np.newaxis])).astype(np.float32,
                                                                        copy=False)
                all_boxes[j][iter_i*batch_size+i] = c_dets
        
        return all_boxes
  

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects


    def get_output_dir(self, name, phase):
        """Return the directory where experimental artifacts are placed.
        If the directory does not exist, it is created.
        A canonical path is built using the name from an imdb and a network
        (if not None).
        """
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir


    def get_voc_results_file_template(self, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + self.set_type + '_%s.txt' % (cls)
        filedir = os.path.join('./tmp/', 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.labelmap):
            if self.display:
                print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.dataset.ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))


    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        aps_50 = []
        aps_50_95 = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        for i, cls in enumerate(self.labelmap):
            filename = self.get_voc_results_file_template(cls)

            r50_95 = []
            for thre in range(50, 96, 5):
                rec, prec, ap = self.voc_eval(detpath=filename,
                                              classname=cls,
                                              cachedir=cachedir,
                                              ovthresh=thre/100,
                                              use_07_metric=use_07_metric
                                              )
                r50_95 += [ap]
                if thre == 50: aps_50 += [ap]

            aps_50_95 += [np.mean(r50_95)]
            print('for {}, ap50 {:.4f}, ap50_95 {:.4f}'.format(cls, aps_50[-1], aps_50_95[-1]))

        if self.display:
            self.map = np.mean(aps_50_95)
            print('Mean AP = {:.4f}'.format(np.mean(aps_50_95)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps_50_95:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps_50_95)))
            # print('~~~~~~~~')
            # print('')
            # print('--------------------------------------------------------------')
            # print('Results computed with the **unofficial** Python eval code.')
            # print('Results should be very close to the official MATLAB eval code.')
            # print('--------------------------------------------------------------')
        else:
            # self.map = np.mean(aps_50)
            self.map = np.mean(aps_50_95)
            self.ap50 = np.mean(aps_50)
            print('AP50: {:.4f} ||  AP50_95: {:.4f}'.format(np.mean(aps_50), np.mean(aps_50_95)))


    def voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def voc_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(self.annopath % (imagename))
                if i % 100 == 0 and self.display:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
            if self.display:
                print('Saving cached annotations to {:s}'.format(cachefile))
            # with open(cachefile, 'wb') as f:
            #     pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()


if __name__ == '__main__':
    pass