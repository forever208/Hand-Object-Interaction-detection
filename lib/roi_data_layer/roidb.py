"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL


"""one example of roidb (labels):"""
#    {'boxes': array([[  3, 446, 317, 675], [275, 425, 524, 632], [  3, 250, 182, 479], [ 21, 526, 810, 714]], dtype=uint16),
#     'gt_classes': array([2, 2, 1, 1], dtype=int32),
#     'gt_ishard': array([0, 0, 0, 0], dtype=int32),
#     'gt_overlaps': <4x3 sparse matrix of type '<class 'numpy.float32'>' with 4 stored elements in Compressed Sparse Row format>,
# 	'flipped': False, 'seg_areas': array([ 72450.,  52000.,  41400., 149310.], dtype=float32),
# 	'contactstate': array([3, 3, 0, 0], dtype=int32),
# 	'contactright': array([0, 0, 1, 0], dtype=int32),
# 	'contactleft': array([0, 0, 0, 1], dtype=int32),
# 	'unitdx': array([-0.3277727 ,  0.17134118,  0.        ,  0.        ], dtype=float32),
# 	'unitdy': array([-0.9447566 ,  0.98521173,  0.        ,  0.        ], dtype=float32),
# 	'magnitude': array([0.20746084, 0.09338094, 0.        , 0.        ], dtype=float32),
# 	'handside': array([1, 0, 0, 0], dtype=int32), 'img_id': 85945,
# 	'image': '/content/Hand-Object-Interaction-detection/data/VOCdevkit2007_handobj_100K/VOC2007/JPEGImages/study_v_37UX5-VFj7Q_frame000338.jpg',
# 	'width': 1280, 'height': 720,
# 	'max_classes': array([2, 2, 1, 1]),
# 	'max_overlaps': array([1., 1., 1., 1.], dtype=float32), 'need_crop': 0}


def prepare_roidb(imdb):
    """
    This function pre-computes the maximum overlap, taken over ground-truth boxes, between each ROI and each gt box.
    The class with maximum overlap is also recorded.
    """

    roidb = imdb.roidb    # labels list [{}, {}, ...], each element is a dict that contains all labels for one image
    if not (imdb.name.startswith('coco')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                 for i in range(imdb.num_images)]

    for i in range(len(imdb.image_index)):
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps

        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)

        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    """
    rank roidb based on the ratio between width and height.
    :param roidb: labels list [{}, {}, ...], each element is a dict that contains all labels for one image
    :return: 1D array [ratio1, ratio2,...], ratio order of all images from small to large
             ratio_index: 1D array [4, 2, 1,...] shows the original index order before sorting
    """
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

    ratio_list = []

    # for each image, judge if it need crop
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
    """
    filter the image without bounding box.
    """

    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0:
            del roidb[i]
            i -= 1
        i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb


def combined_roidb(imdb_names, training=True, leftright=False):
    """
    Combine multiple roidbs
    :param imdb_names: string, 'voc_2007_trainval'
    :return imdb: a pascal_voc() class instance
            roidb: labels list [{}, {}, ...], each element is a dict that contains all labels for one image
            ratio_list: 1D array [ratio1, ratio2,...], ratio order of all images from small to large
            ratio_index: 1D array [4, 2, 1,...] shows the original index order before sorting
    """

    def get_training_roidb(imdb, leftright=False):
        """
        :param imdb: a pascal_voc() instance
        :return: labels list [{}, {}, ...], each element is a dict that contains all labels for one image
        """
        if cfg.TRAIN.USE_FLIPPED:
            if leftright:
                print('Appending horizontally-flipped training examples...')
                imdb.append_flipped_images()
                print('done')
            else:
                print('Appending horizontally-flipped training examples...')
                imdb.append_flipped_images(leftright)
                print('done')

        print('Preparing training data...')
        prepare_roidb(imdb)
        # ratio_index = rank_roidb_ratio(imdb)
        print('done')
        return imdb.roidb


    def get_roidb(imdb_name):
        """
        :param imdb_name: string, 'voc_2007_trainval'
        :return: labels list [{}, {}, ...], each element is a dict that contains all labels for one image
        """
        imdb = get_imdb(imdb_name)    # return a pascal_voc() class instance which contains all annotations
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        print('dataset name: ', imdb_name)
        roidb = get_training_roidb(imdb)
        return roidb


    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)    # get a pascal_voc() class instance

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)
    return imdb, roidb, ratio_list, ratio_index
