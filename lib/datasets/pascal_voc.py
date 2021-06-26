from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


import os
import numpy as np
import scipy.sparse
import subprocess
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval, voc_eval_hand

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg


class pascal_voc(imdb):

    def __init__(self, image_set, year, devkit_path=None):
        """
        :param image_set: string, 'train' or 'val' or 'trainval' or 'test'
        :param year: string, '2007'
        :param devkit_path:
        """
        imdb.__init__(self, 'voc_'+year+'_'+image_set)   # equal to super().__init__(para1，para2, ...)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path    # 'data/VOCdevkit2007_handobj_100K'
        self._data_path = os.path.join(self._devkit_path, 'VOC'+self._year)    # 'data/VOCdevkit2007_handobj_100K/VOC2007'

        self._classes = ('__background__', 'targetobject', 'hand')    # rewrite this attribute of parent class
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))    # {'__background__':0, 'targetobject':1, 'hand':2}
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()    # img filenames without .jpg ['boardgame_v_-4m5TwI-698_frame000134', ...]

        # self._roidb_handler = self.selective_search_roidb    # Default to roidb handler
        self._roidb_handler = self.gt_roidb    # labels list [{}, {}, ...], each element is a dict that contains all labels for one image
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), 'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])


    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i


    def image_path_from_index(self, index):
        """
        :param index: 'boardgame_v_-4m5TwI-698_frame000134'
        :return: image path, 'data/VOCdevkit2007_handobj_100K/VOC2007/JPEGImages/boardgame_v_-4m5TwI-698_frame000134.jpg'
        """
        image_path = os.path.join(self._data_path, 'JPEGImages', index+self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path


    def _load_image_set_index(self):
        """
        Load the image filenames based on the file: 'data/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/val.txt')
        :return: a list of all image filenames (without postfix .jpg), ['boardgame_v_-4m5TwI-698_frame000134', ...]
        """

        # image_set_file: "data/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/val.txt"
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main', self._image_set+'.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]

        return image_index


    def _get_default_path(self):
        """
        :return: the default path of PASCAL_VOC dataset, 'data/VOCdevkit2007_handobj_100K'
        """
        default_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year + '_handobj_100K')
        print(f'--------> dataset path = {default_path}')
        return default_path


    def gt_roidb(self):
        """
        This function saves a dataset cache file to speed up future calls.
        :return: labels list [{}, {}, ...], each element is a dict that contains all labels for one image
        """

        # cache_file: absolute dir '/.../data/cache_handobj_100K/VOC_2007_trainval_gt_roidb.pkl'
        cache_file = os.path.join(self.cache_path, self.name+'_gt_roidb.pkl')

        # load sequenced data from .pkl cache file
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt annotations loaded from {}'.format(self.name, cache_file))
            return roidb

        # save the annotation (parsed from xml) into .pkl cache file
        # [{}, {}, ...] each element is a dictionary that contains all labels for one image
        gt_roidb = [self._load_pascal_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name+'_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss annotations loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)

        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb


    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb


    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _is_not_legimate(self, ele):
        return (ele == None or ele.text == 'None' or ele.text == None)


    """core function that parse the xml annotations"""
    def _load_pascal_annotation(self, index):
        """
        given an xml file (PASCAL VOC format), parse the ground truth info
        :param index: string, an image filename (without .jpg), e.g. 'boardgame_v_-4m5TwI-698_frame000134'
        :return: a dictionary of gt labels
        """

        # filename: 'data/VOCdevkit2007_handobj_100K/VOC2007/Annotations/boardgame_v_-4m5TwI-698_frame000134.xml'
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        # initialise the ground truth info as zero np array
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)    # "Seg" area in pascal is just the rectangle area of bbox
        ishards = np.zeros((num_objs), dtype=np.int32)
        contactstate = np.zeros((num_objs), dtype=np.int32)
        contactright = np.zeros((num_objs), dtype=np.int32)
        contactleft = np.zeros((num_objs), dtype=np.int32)
        magnitude = np.zeros((num_objs), dtype=np.float32)
        unitdx = np.zeros((num_objs), dtype=np.float32)
        unitdy = np.zeros((num_objs), dtype=np.float32)
        handside = np.zeros((num_objs), dtype=np.int32)

        # Load gt labels for every object (hand or target_object)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text) - 1, 0)
            y1 = max(float(bbox.find('ymin').text) - 1, 0)
            x2 = max(float(bbox.find('xmax').text) - 1, 0)
            y2 = max(float(bbox.find('ymax').text) - 1, 0)

            # labels that we don't care too much
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]    # cls = 1 or 2
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            hs = obj.find('contactstate')
            hs = 0 if self._is_not_legimate(hs) else int(hs.text)
            contactstate[ix] = hs

            contactr = obj.find('contactright')
            contactr = 0 if self._is_not_legimate(contactr) else int(contactr.text)
            contactright[ix] = contactr

            contactl = obj.find('contactleft')
            contactl = 0 if self._is_not_legimate(contactl) else int(contactl.text)
            contactleft[ix] = contactl

            mag = obj.find('magnitude')
            mag = 0 if self._is_not_legimate(mag) else float(mag.text) * 0.001  # balance scale
            magnitude[ix] = mag

            # n_n = obj.find('normalizednorm')
            # n_n = 0 if self._is_not_legimate(n_n) else float(n_n.text)
            # n_norm[ix] = n_n

            dx = obj.find('unitdx')
            dx = 0 if self._is_not_legimate(dx) else float(dx.text)
            unitdx[ix] = dx

            dy = obj.find('unitdy')
            dy = 0 if self._is_not_legimate(dy) else float(dy.text)
            unitdy[ix] = dy

            lr = obj.find('handside')
            lr = 0 if self._is_not_legimate(lr) else float(lr.text)
            handside[ix] = lr

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'contactstate': contactstate,
                'contactright': contactright,
                'contactleft': contactleft,
                'unitdx': unitdx,
                'unitdy': unitdy,
                # 'normalizednorm': n_norm,
                'magnitude': magnitude,
                'handside': handside}


    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id


    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def _write_voc_results_file(self, all_boxes):
        """
        write detection results into "data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_targetobject.txt"
        :param all_boxes: 2D list, 3 rows, num_images columns, each element is a 2D array (num_bbox, 11)
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)

            # each class has an individual file
            with open(filename, 'wt') as f:

                # write current class predictions of all images into a file
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]    # get current class predictions of one image
                    if dets == []:
                        continue

                    # write current class predictions of one image, each row is a prediction
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.
                                format(index, dets[k, 4],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1,
                                       int(dets[k, 5]), dets[k, 6], dets[k, 7], dets[k, 8], dets[k, 9], dets[k, 10]))


    def _do_python_eval(self, output_dir='output'):
        """
        Do AP evaluation
        :param output_dir: output/res101/voc_2007_test/hand0bj_100K/
        """

        # data/VOCdevkit2007_handobj_100K/VOC2007/Annotations/{:s}.xml
        annopath = os.path.join(self._devkit_path, 'VOC'+self._year, 'Annotations', '{:s}.xml')

        # data/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test.txt
        imagesetfile = os.path.join(self._devkit_path, 'VOC'+self._year, 'ImageSets', 'Main', self._image_set+'.txt')

        # data/VOCdevkit2007_handobj_100K/annotations_cache
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')

        use_07_metric = True if int(self._year) < 2010 else False    # The PASCAL VOC metric changed in 2010
        print('VOC07 metric? ' + ('--> Yes' if use_07_metric else '--> No'))

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue

            # data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_targetobject.txt
            # data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_hand.txt
            filename = self._get_voc_results_file_template().format(cls)    # filename of the saved detections

            # hand, target AP evaluation
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            print('AP for {} = {:.4f}'.format(cls, ap))

            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

            # hand + x, AP evaluation
            if cls == 'hand':
                filename = self._get_voc_results_file_template()  # .format(cls)
                for constraint in ['handstate', 'handside', 'objectbbox', 'all']:
                    rec, prec, ap = voc_eval_hand(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                                                  use_07_metric=use_07_metric, constraint=constraint)
                    print('AP for {} + {} = {:.4f}'.format(cls, constraint, ap))
                    with open(os.path.join(output_dir, cls + f'_pr_{constraint}.pkl'), 'wb') as f:
                        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)


    def evaluate_detections(self, all_boxes, output_dir):
        """
        write detection results into txt file
        evaluate the AP
        :param all_boxes: 2D list, 3 rows, num_images columns, each element is a 2D array (num_bbox, 11)
        :param output_dir: output/res101/voc_2007_test/hand0bj_100K/
        """

        # 1. write detection results into "data/VOCdevkit2007_handobj_100K/results/VOC2007/Main/comp4_det_test_targetobject.txt"
        self._write_voc_results_file(all_boxes)

        # 2. AP evaluation
        self._do_python_eval(output_dir)

        # NO execution when competition_mode is on
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;
    embed()
