# --------------------------------------------------------
# redefine the number of training and test set
# training set < 89916, test set < 9913
# Written by Mang Ning
# --------------------------------------------------------


import argparse


def main():
    # choose the numbet of training images and test images
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--train', dest='train',
                        help='number of training images',
                        default='19695', type=int)  # 3 senarios (boardgame, diy, drink)
    parser.add_argument('--test', dest='test',
                        help='number of test images',
                        default='1666', type=int)  # 3 senarios (boardgame, diy, drink)
    args = parser.parse_args()
    num_training = args.train
    num_test = args.test

    with open('pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/trainval.txt', 'r') as r:
      lines=r.readlines()

    with open('pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/trainval_cp.txt', 'w') as w:
      for ind, l in enumerate(lines):
        w.write(l)
        if ind == (num_training-1):
          break


    with open('pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test.txt', 'r') as r:
      lines=r.readlines()

    with open('pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test_cp.txt', 'w') as w:
      for ind, l in enumerate(lines):
        w.write(l)
        if ind == (num_test-1):
          break


if __name__ == '__main__':
    main()

