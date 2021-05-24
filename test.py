import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math 
import matplotlib
import time
#import py_camer
reload(sys)  
sys.setdefaultencoding('utf-8')  
 
CLASSES = ('__background__',
            'person')
 
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_iter_2000.caffemodel')}
 
#******************    把检测结果标注在视频上   ************************
def gain_box_score(im, class_name, dets, start_time,time_takes,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]  #返回置信度大于阈值的窗口下标  
    if  len(inds) == 0:
        cv2.imshow("视频检测",im)
    else:
                                #cv 读入的是BGR
        im = im[:, :, (2, 1, 0)]#check 通道顺序 换成RGB
        zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/ukai.ttc") #字体  
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i in inds:
            bbox = dets[i, :4]    #坐标位置（Xmin,Ymin,Xmax,Ymax）
            score = dets[i, -1]   #置信度得分
    #- -------------------  开始画框 -----------------------      
            x1 = bbox[0]
            y1 = bbox[1] 
            x2 = bbox[2] 
            y2 = bbox[3]
            im = im.astype('uint8')
            im = im[:,:,[2,1,0]]
            im = im.copy()
            end_time = time.time()
            current_time = time.ctime() #获得当前系统时间
            fps = round(1/(end_time - start_time),2)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3) # 用矩形圈出人
            cv2.putText(im, str(round(score,2)), (int(x1+x2)/2,int(y2-y2/20)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) #显示物体的得分
            cv2.putText(im, str(class_name), (x1,int(y2-y2/20)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) #显示物体的类别
            cv2.putText(im, str(current_time), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) #显示时间
            
            cv2.putText(im, "fps:"+str(fps), (30,60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) #显示帧率
            cv2.putText(im, "takes time :"+str(round(time_takes*1000,1))+"ms", (30,90),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) #显示检测时间
            cv2.imshow("视频检测",im)
     
#====================================  结束  ==============================
 
def demo(net, image_name, start_time):
    """Detect object classes in an image using pre-computed object proposals."""
    im = image_name
 
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)  #检测，返回得分和目标区域所在位置
    timer.toc()
    time_takes = timer.total_time
    #print ('Detection took {:.3f}s for '
          # '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    # Visualize detections for each class
    CONF_THRESH = 0.75
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
 
 #  ----------------------  在视频上进行标注 -----------------------    
        gain_box_score(im, cls, dets, start_time, time_takes, thresh=CONF_THRESH)
 
      
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')
 
    args = parser.parse_args()
 
    return args
 
if __name__ == '__main__':
    
 
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
 
    args = parse_args()
 
    prototxt =  '/home/hito/py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt'
    caffemodel = '/home/hito/py-faster-rcnn/data/faster_rcnn_models/ZF_faster_rcnn_iter_2000.caffemodel' 
 
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
 
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
 
    print '\n\nLoaded network {:s}'.format(caffemodel)
 
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
 
# -------------------  摄像头模块 ----------------
    
    cap=cv2.VideoCapture(0)  #打开0号摄像头
    while(1):
        ret, frame = cap.read()    # show a frame        
        start_time = time.time()
        demo(net, frame, start_time)
  
        key = cv2.waitKey(1)
        if  key == ord('q') or key == ord('Q') or key == 27 :     # ESC:27  key: quit program    

————————————————
版权声明：本文为CSDN博主「火腿肠的幸福生活」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_33202928/article/details/82216898