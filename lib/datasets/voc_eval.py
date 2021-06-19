# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os, sys, pdb, math
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# sys.path.append('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/lib/model/utils')
# from lib.datasets.viz_hand_obj_debug import *

def parse_rec(filename):
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

        obj_struct['handstate'] = 0 if obj.find('contactstate').text is None else int(obj.find('contactstate').text)
        obj_struct['leftright'] = 0 if obj.find('handside').text is None else int(obj.find('handside').text)

        obj_struct['objxmin'] = None if obj.find('objxmin').text in [None, 'None'] else float(obj.find('objxmin').text)
        obj_struct['objymin'] = None if obj.find('objymin').text in [None, 'None'] else float(obj.find('objymin').text)
        obj_struct['objxmax'] = None if obj.find('objxmax').text in [None, 'None'] else float(obj.find('objxmax').text)
        obj_struct['objymax'] = None if obj.find('objymax').text in [None, 'None'] else float(obj.find('objymax').text)

        if obj_struct['objxmin'] is not None and obj_struct['objymin'] is not None and obj_struct[
            'objxmax'] is not None and obj_struct['objymax'] is not None:
            obj_struct['objectbbox'] = [obj_struct['objxmin'], obj_struct['objymin'], obj_struct['objxmax'],
                                        obj_struct['objymax']]
        else:
            obj_struct['objectbbox'] = None

        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the VOC 07 11 point method (default:False).
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


'''
@description: raw evaluation for fasterrcnn
'''


def voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=False):
    """
    PASCAL VOC AP evaluation.
    :param detpath: detection path, "data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_hand.txt"
    :param annopath: gt lables path, "data/VOCdevkit2007_handobj_100K/VOC2007/Annotations/{:s}.xml"
    :param imagesetfile: image filename, one image per line. "data/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test.txt"
    :param classname: 'targetobject', 'hand'
    :param cachedir: annotation cash dir, "data/VOCdevkit2007_handobj_100K/annotations_cache"
    :param ovthresh: Overlap threshold (default = 0.5)
    :param use_07_metric: Whether to use VOC07's 11 point AP computation
    :return:
    """

    print('\n IoU threshold of AP evaluation for {} = {} \n'.format(classname, ovthresh))

    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)

    # data/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test.txt_annots.pkl
    cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)

    # read image filenames
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # 1. load, parse and save gt labels (pkl file) based on image filename
    if not os.path.isfile(cachefile):
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))    # annopath.format(imagename), replace {:s} with imagename
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i+1, len(imagenames)))
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    # load gt labels (pkl file)
    else:
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # 2. extract gt labels for current class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'].lower() == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # 3. read file of detection results
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:2 + 4]] for x in splitlines])

    # 4. AP calculations
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
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
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

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

    # compute precision and recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(recall, precision, use_07_metric)

    return recall, precision, ap


'''
@description: eval hand-object-interaction
@compare: hand_bbox, object_bbox, state, side
TODO:
(1) prepare gt and det of hand --> (image_path, score, handbbox, state, side, objectbbox)
'''


def voc_eval_hand(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=False, constraint=''):
    """
    AP evaluation for hand interaction
    :param detpath: detection results path, "data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_hand.txt"
    :param annopath: gt lables path, "data/VOCdevkit2007_handobj_100K/VOC2007/Annotations/{:s}.xml"
    :param imagesetfile: image filename, one image per line. "data/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test.txt"
    :param classname: 'hand'
    :param cachedir: annotation cash dir, "data/VOCdevkit2007_handobj_100K/annotations_cache"
    :param ovthresh: Overlap threshold (default = 0.5)
    :param use_07_metric: Whether to use VOC07's 11 point AP computation
    :param constraint: one of ['handstate', 'handside', 'objectbbox', 'all']
    """

    print(f'\n\n*** current overlap thd = {ovthresh}')
    print(f'*** current constraint = {constraint}')
    assert constraint in ['', 'handstate', 'handside', 'objectbbox', 'all']

    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    # data/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test.txt_annots.pkl
    cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)  # cachefile = test.txt_annots.pkl

    # read image filenames
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # 1. load, parse and save gt labels (pkl file) based on image filename
    if not os.path.isfile(cachefile):
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    # load gt labels (pkl file)
    else:
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    """
    recs is a dictionary of gt labels, each key is an image name, each value is a list of gt labels
    e.g. recs['boardgame_v_-22f4DmhjLs_frame000022'] is shown as follows:
    {'boardgame_v_-22f4DmhjLs_frame000022': [{'name': 'hand', 'difficult': 0, 'bbox': [851, 508, 900, 542], 'handstate': 0, 'leftright': 0, 'objxmin': None, 'objymin': None, 'objxmax': None, 'objymax': None, 'objectbbox': None},
                                             {'name': 'hand', 'difficult': 0, 'bbox': [266, 226, 332, 254], 'handstate': 3, 'leftright': 0, 'objxmin': 238.0, 'objymin': 240.0, 'objxmax': 290.0, 'objymax': 266.0, 'objectbbox': [238.0, 240.0, 290.0, 266.0]}]
    """

    # 2. extract gt labels for hand class
    class_recs = {}
    npos = 0
    for imagename in imagenames:    # for each image

        # R: each element is a dictionary of a hand labels
        # [{'name': 'hand', 'difficult': 0, 'bbox': [851, 508, 900, 542], 'handstate': 0, 'leftright': 0}, {}, ...{}]
        R = [obj for obj in recs[imagename] if obj['name'].lower() == classname]
        bbox = np.array([x['bbox'] for x in R])    # 2D int array, each row is a bbox
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)    # 2D bool array, each row is a True/False
        handstate = np.array([x['handstate'] for x in R]).astype(np.int)    # 2D int array, each row is 0/1/2/3/4
        leftright = np.array([x['leftright'] for x in R]).astype(np.int)    # 2D int array, each row is 0/1
        objectbbox = np.array([x['objectbbox'] for x in R])    # 2D float array, each row is a target bbox
        det = [False] * len(R)
        npos = npos + sum(~difficult)    # number of non-difficult gt hand bbox for all images

        # pack all useful gt info into a dictionary, each key-value is an image
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'handstate': handstate,
                                 'leftright': leftright,
                                 'objectbbox': objectbbox,
                                 'det': det}

    # 3. read detection results (hand / target)
    # BB_det_object: 2D list, each element is a target detection [cls_score x1 y1 x2 y2 contactstate magnitude dx dy handside 1]
    BB_det_object, image_ids_object, detfile_object = extract_BB(detpath, extract_class='targetobject')
    BB_det_hand, image_ids_hand, detfile_hand = extract_BB(detpath, extract_class='hand')
    ho_dict = make_hand_object_dict(BB_det_object, BB_det_hand, image_ids_object, image_ids_hand)

    # hand detection results, 2D list, each sub-list has 8 elements
    # [img_filename, handscore, handbbox, contactstate, vector, side, objectbbox, objectbbox_score]
    # ['boardgame_v_-22f4DmhjLs_frame000022', 0.794, array([846.9, 504.3, 890.8, 545.2]), 1.0, array([ 0.071, -0.023, -0.097]), 0.0, [747.6, 152.5, 1135.1, 701.1], 0.875]
    hand_det_res = gen_det_result(ho_dict)

    image_ids = [x[0] for x in hand_det_res]    # image filename
    confidence = np.array([x[1] for x in hand_det_res])    # hand class score
    BB_det = np.array([x[2] for x in hand_det_res]).astype(float)    # hand bbox, 2D array
    handstate_det = np.array([int(x[3]) for x in hand_det_res])  # contact state
    leftright_det = np.array([int(x[5]) for x in hand_det_res])  # hand sie
    objectbbox_det = [x[6] for x in hand_det_res]    # target bbox, 2D list
    objectbbox_score_det = [x[7] for x in hand_det_res]    # target score

    nd = len(image_ids)    # number of test images
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB_det.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)

        # ======== det ======== #
        image_ids = [image_ids[x] for x in sorted_ind]
        confidence_det = [confidence[x] for x in sorted_ind]
        BB_det = BB_det[sorted_ind, :]
        handstate_det = handstate_det[sorted_ind]
        leftright_det = leftright_det[sorted_ind]
        objectbbox_det = [objectbbox_det[x] for x in sorted_ind]  # objectbbox_det[sorted_ind, :]
        objectbbox_score_det = [objectbbox_score_det[x] for x in sorted_ind]  # objectbbox_det[sorted_ind, :]
        # ============================= #

        # go down dets and mark TPs and FPs
        for d in range(nd):
            # det
            image_id_det = image_ids[d]
            score_det = confidence_det[d]
            bb_det = BB_det[d, :].astype(float)
            hstate_det = handstate_det[d].astype(int)
            hside_det = leftright_det[d].astype(int)
            objbbox_det = objectbbox_det[d]  # .astype(float)
            objbbox_score_det = objectbbox_score_det[d]
            # print(f'debug hand-obj: {bb_det} {objbbox_det}')

            # gt
            ovmax = -np.inf
            R = class_recs[image_ids[d]]
            BBGT = R['bbox'].astype(float)
            hstate_GT = R['handstate'].astype(int)
            hside_GT = R['leftright'].astype(int)
            objbbox_GT = R['objectbbox']  # .astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb_det[0])
                iymin = np.maximum(BBGT[:, 1], bb_det[1])
                ixmax = np.minimum(BBGT[:, 2], bb_det[2])
                iymax = np.minimum(BBGT[:, 3], bb_det[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb_det[2] - bb_det[0] + 1.) * (bb_det[3] - bb_det[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if constraint == '':
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:  # add diff constraints here for diff eval
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            elif constraint == 'handstate':
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax] and hstate_GT[jmax] == hstate_det:  # add diff constraints here for diff eval
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            elif constraint == 'handside':
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax] and hside_GT[jmax] == hside_det:  # add diff constraints here for diff eval
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            elif constraint == 'objectbbox':
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax] and val_objectbbox(objbbox_GT[jmax], objbbox_det, image_ids[d]):  # add diff constraints here for diff eval
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            elif constraint == 'all':
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax] and hstate_GT[jmax] == hstate_det and hside_GT[jmax] == hside_det and val_objectbbox(objbbox_GT[jmax], objbbox_det, image_ids[d]):  # add diff constraints here for diff eval
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(recall, precision, use_07_metric)

    return recall, precision, ap


# ======== auxiluary functions ======== #
def val_objectbbox(objbbox_GT, objbbox_det, imagepath, threshold=0.5):
    if objbbox_GT is None and objbbox_det is None:
        # print('None - None')
        return True
    elif objbbox_GT is not None and objbbox_det is not None:
        if get_iou(objbbox_GT, objbbox_det) > threshold:
            # print('Yes', get_iou(objbbox_GT, objbbox_det), objbbox_GT, objbbox_det, imagepath)
            return True
        # else:
        # print('No', get_iou(objbbox_GT, objbbox_det), objbbox_GT, objbbox_det, imagepath)

    else:
        # print(f'None - Float')
        False


def get_iou(bb1, bb2):
    assert (bb1[0] <= bb1[2] and bb1[1] <= bb1[3] and bb2[0] <= bb2[2] and bb2[1] <= bb2[3]), print(bb1, bb2)

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def extract_BB(detpath, extract_class):
    """
    read detection results file (.txt)
    :param detpath: detection results path, "data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_hand.txt"
    :param extract_class: "hand" or "targetobject"
    :return:
        detfile: "data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_targetobject.txt"
        BB: 2D list, each element is a bbox detection [cls_score x1 y1 x2 y2 contactstate magnitude dx dy handside 1]
        image_ids: a list image filename, each element corresponds to a bbox detection
    """

    # read detection results
    detfile = detpath.format(extract_class)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # lines: ['boardgame_v_-22f4DmhjLs_frame000022 0.875 747.6 152.5 1135.1 701.1 0.0 0.083 -0.090 -0.044 0.000 1.000\n', ...]
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    BB = np.array([[float(z) for z in x[1:]] for x in splitlines])

    return BB, image_ids, detfile


def make_hand_object_dict(BB_o, BB_h, image_o, image_h):
    """
    for a image, zip target and hand predictions into a dictionary
    :param BB_o: 2D list, each element is a target detection [cls_score x1 y1 x2 y2 contactstate magnitude dx dy handside 1]
    :param BB_h: 2D list, each element is a hand detection [cls_score x1 y1 x2 y2 contactstate magnitude dx dy handside 1]
    :param image_o: a list image filename, each element corresponds to a target detection
    :param image_h: a list image filename, each element corresponds to a hand detection
    :return:
        ho_dict{'boardgame_v_-22f4DmhjLs_frame000022': {'hands': [array([ 9.940e-01,  1.219e+03,  3.674e+02,  1.272e+03,  5.052e+02, 0.000e+00,  9.200e-02, -7.200e-02,  7.000e-02,  1.000e+00, 1.000e+00]),
                                                                  array([ 8.190e-01,  1.633e+02,  4.764e+02,  2.049e+02,  5.400e+02, 0.000e+00,  1.320e-01, -8.300e-02, -5.600e-02,  0.000e+00, 1.000e+00]),
                                                                  array([ 7.940e-01,  8.469e+02,  5.043e+02,  8.908e+02,  5.452e+02, 1.000e+00,  7.100e-02, -2.300e-02, -9.700e-02,  0.000e+00, 1.000e+00]),
                                                                  array([ 6.580e-01,  9.093e+02,  6.196e+02,  9.682e+02,  7.175e+02, 0.000e+00,  7.500e-02, -1.400e-02, -9.900e-02,  0.000e+00, 1.000e+00])
                                                                  ],
                                                       'objects': [array([ 8.750e-01,  7.476e+02,  1.525e+02,  1.135e+03,  7.011e+02, 0.000e+00,  8.300e-02, -9.000e-02, -4.400e-02,  0.000e+00, 1.000e+00]),
                                                                   array([ 5.850e-01,  1.490e+01,  2.489e+02,  2.247e+02,  7.152e+02, 0.000e+00,  8.300e-02, -7.000e-02, -7.100e-02,  1.000e+00, 1.000e+00])
                                                                  ]
                                                       }
                'boardgame_v_-22f4DmhjLs_frame000023': {
                                                        }}
    """
    ho_dict = {}
    for bb_h, id_h in zip(BB_h, image_h):
        if id_h in ho_dict:
            ho_dict[id_h]['hands'].append(bb_h)
        else:
            ho_dict[id_h] = {'hands': [bb_h], 'objects': []}

    for bb_o, id_o in zip(BB_o, image_o):
        if id_o in ho_dict:
            ho_dict[id_o]['objects'].append(bb_o)
        else:
            ho_dict[id_o] = {'hands': [], 'objects': [bb_o]}

    return ho_dict


def calculate_center(bb):
    return [(bb[1] + bb[3]) / 2, (bb[2] + bb[4]) / 2]


'''
@description: 
[image_path, hand_score, hand_bbox, state, vector, side, objectbbox, object_score]
'''


def gen_det_result(ho_dict):
    # take all results
    hand_det_res = []

    for key, info in ho_dict.items():
        object_cc_list = []
        object_bb_list = []
        object_score_list = []

        for j, object_info in enumerate(info['objects']):
            object_bbox = [object_info[1], object_info[2], object_info[3], object_info[4]]
            object_cc_list.append(calculate_center(object_info))  # is it wrong???
            object_bb_list.append(object_bbox)
            object_score_list.append(float(object_info[0]))
        object_cc_list = np.array(object_cc_list)

        for i, hand_info in enumerate(info['hands']):
            hand_path = key
            hand_score = hand_info[0]
            hand_bbox = hand_info[1:5]
            hand_state = hand_info[5]
            hand_vector = hand_info[6:9]
            hand_side = hand_info[9]

            if hand_state <= 0 or len(object_cc_list) == 0:
                to_add = [hand_path, hand_score, hand_bbox, hand_state, hand_vector, hand_side, None, None]
                hand_det_res.append(to_add)
            else:
                hand_cc = np.array(calculate_center(hand_info))
                point_cc = np.array([(hand_cc[0] + hand_info[6] * 10000 * hand_info[7]),
                                     (hand_cc[1] + hand_info[6] * 10000 * hand_info[8])])
                dist = np.sum((object_cc_list - point_cc) ** 2, axis=1)

                dist_min = np.argmin(dist)
                # get object bbox
                target_object_score = object_score_list[dist_min]
                #
                target_object_bbox = object_bb_list[dist_min]
                to_add = [hand_path, hand_score, hand_bbox, hand_state, hand_vector, hand_side, target_object_bbox,
                          target_object_score]
                hand_det_res.append(to_add)

    return hand_det_res


# ======== debug ======== #
def debug_det_gt(image_name, det_info, gt_info, d):
    os.makedirs('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/images/debug', exist_ok=True)
    # det_info = [bb_det, hstate_det, hside_det, objbbox_det, score_det， objbbox_score_det]
    # gt_info = [BBGT[jmax], hstate_GT[jmax], hside_GT[jmax], objbbox_GT[jmax]]

    genre, vid_folder = image_name.split('_', 1)[0], image_name.split('_', 1)[1][:13]
    genre_name = f'{genre}_videos'
    image_path = os.path.join('/y/jiaqig/hand_cache', genre_name, vid_folder, image_name + '.jpg')
    image = Image.open(image_path).convert("RGBA")

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/lib/model/utils/times_b.ttf',
                              size=20)
    width, height = image.size

    # ======== plot det ======== #

    hand_bbox_det = list(det_info[0])
    hand_bbox_det = list(int(np.round(x)) for x in hand_bbox_det)
    image = draw_hand_mask(image, draw, 0, hand_bbox_det, det_info[4], det_info[2], det_info[1], width, height, font)

    if det_info[3] is not None:
        object_bbox_det = list(det_info[3])
        object_bbox_det = list(int(np.round(x)) for x in object_bbox_det)
        image = draw_obj_mask(image, draw, 0, object_bbox_det, det_info[5], width, height, font)

        if det_info[1] > 0:  # in contact hand

            obj_cc, hand_cc = calculate_center_PIL(hand_bbox_det), calculate_center_PIL(object_bbox_det)
            draw_line_point(draw, 0, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))

    # ======== plot gt ======== #

    hand_bbox_gt = list(gt_info[0])
    hand_bbox_gt = list(int(np.round(x)) for x in hand_bbox_gt)
    image = draw_hand_mask(image, draw, 1, hand_bbox_gt, 1.0, gt_info[2], gt_info[1], width, height, font)

    if gt_info[3] is not None:
        object_bbox_gt = list(gt_info[3])
        object_bbox_gt = list(int(np.round(x)) for x in object_bbox_gt)
        image = draw_obj_mask(image, draw, 1, object_bbox_gt, 1.0, width, height, font)

        if gt_info[1] > 0:  # in contact hand

            obj_cc, hand_cc = calculate_center_PIL(hand_bbox_gt), calculate_center_PIL(object_bbox_gt)
            draw_line_point(draw, 1, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))

    # ======== save ======== #

    save_name = image_name + f'_draw_{d:04d}.png'
    image.save(os.path.join('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/images/debug', save_name))