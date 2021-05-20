

## Introduction

This repo is based on Faster-RCNN.

Running the repo in Colab is recommended, copy the file [HOI detection.ipynb](https://colab.research.google.com/drive/1AEAgVhDKNsmUmmDRG9dP9y4VToC4pVn1?usp=sharing), then run it on Colab. (remember to change the runtime type to GPU in Colab)




## 【1】Prerequisites (Colab user can skip this step) 

* Python 3.7
* Pytorch 1.6
* CUDA 10.1

Create a new conda called handobj, install pytorch-1.6.0
```
conda create --name handobj python=3.7
conda activate handobj

# for cpu
conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch  

# for GPU
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```


## 【2】Installation & Compile

Clone the code
```
git clone https://github.com/forever208/Hand-Object-Interaction-detection.git
```

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:
```
cd lib
python setup.py build develop
cd ..
```

Install coco_python_API
```
mkdir data
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
cd ../../..
```



## 【3】Run Demo

### Download the model
Creat a folder `./models/res101_handobj_100K/pascal_voc`, then download the model.
```
mkdir -p ./models/res101_handobj_100K/pascal_voc
cd models/res101_handobj_100K/pascal_voc
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=166IM6CXA32f9L6V7-EMd9m8gin6TFpim' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=166IM6CXA32f9L6V7-EMd9m8gin6TFpim" -O faster_rcnn_1_8_89999.pth && rm -rf /tmp/cookies.txt
cd ../../..
```

the folder structure looks like this:
```
models
└── res101_handobj_100K
    └── pascal_voc
        └── faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth
```

### run demo

Put your images in the `images/` folder and run the command. 
```
python demo.py --checkepoch=8 --checkpoint=89999
```

A new folder `images_det/` will be created with the detected results



**Params to save detected results** in demo.py you may need for your task:
* hand_dets: detected results for hands, [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
* obj_dets: detected results for object, [boxes(4), score(1), <em>state(1), offset_vector(3), left/right(1)</em>]

We did **not** train the contact_state, offset_vector and hand_side part for objects. We keep them just to make the data format consistent. So, only use the bbox and confidence score infomation for objects.  

**Matching**:

Check the additional [matching.py](https://github.com/ddshan/Hand_Object_Detector/blob/master/lib/model/utils/matching.py) script to match the detection results, **hand_dets** and **obj_dets**, if needed.  


### One Image Demo Output:

Color definitions:
* yellow: object bbox
* red: right hand bbox
* blue: left hand bbox

Label definitions:
* L: left hand
* R: right hand
* N: no contact
* S: self contact
* O: other person contact
* P: portable object contact
* F: stationary object contact (e.g.furniture)

![demo_sample](assets/boardgame_848_sU8S98MT1Mo_00013957.png)




## 【4】Train

### Download dataset
creat a folder `./data`, then download the dataset and unzip it.
```
mkdir data
cd data

wget https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/pascal_voc_format.zip
unzip pascal_voc_format.zip
rm -rf pascal_voc_format.zip

cd ..
mv data/pascal_voc_format/VOCdevkit2007_handobj_100K/ data/
```

### Download pre-trained model
Download pretrained Resnet model by running the command
```
cd data
mkdir pretrained_model
cd pretrained_model

# download the backbone of resnet101
wget https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth
mv resnet101-5d3b4d8f.pth resnet101.pth

# download the backbone of resnet50
wget https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth
mv resnet50-19c8e357.pth resnet50.pth

# download the backbone of resnet152
wget https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth
mv resnet152-b121ed2d.pth resnet152.pth
```

So far, the data/ folder should be like this:
```
data/
├── pretrained_model
│   └── resnet101_caffe.pth
├── VOCdevkit2007_handobj_100K
│   └── VOC2007
│       ├── Annotations
│       │   └── *.xml
│       ├── ImageSets
│       │   └── Main
│       │       └── *.txt
│       └── JPEGImages
│           └── *.jpg
```

To train a hand object detector model with resnet101 on pascal_voc format data, run:
note that, we only support batch_size = 1 right now
```
python trainval_net_Colab.py --bs 1 --nw 2 --net res101 --lr 1e-3 --lr_decay_step 3 --epoch=10 --cuda --use_tfb
```



## 【5】Test
To evaluate the detection performance, run:
```
python test_net_Colab.py --model_name=handobj_100k --save_name=handobj_100k --cuda --checkepoch=8 --checkpoint=89999
```


## 【6】Benchmarking (AP@50)

### Note that
Considering the time consumption, we only use 3 image senarios (boardgame, diy, drink) for the benchmark.

The benchmark is trained with 19695 images and tested with 1666 images. 

The easiest way of doing so is directly modify the files:
```
`data/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/trainval.txt`
`data/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/ImageSets/Main/test.txt`
```



### Comparison of backbones

parameter setting
  - lr_start = 1e-3, lr_end = 1e-6
  - lr_decay_epoch = 3, epochs = 10

| Backbone  | Hand  | Target | Hand + <br>Side | Hand + <br>Contact State | Hand + <br>Target | Hand + <br>All | Model link     |
|-----------|-------|--------|-----------------|--------------------------|-------------------|----------------|----------------|
| ResNet50  | 80.86 | 46.60  | 67.14           | 52.32                    | 26.91             | 20.90          | 1_10_19694.pth |
| ResNet101 | 81.35 | 52.61  | 68.51           | 55.50                    | 30.85             | 28.18          | 1_10_19694.pth |
| ResNet152 | 81.36 | 55.39  | 76.11           | 61.88                    | 37.90             | 29.95          | 1_9_19694.pth  |


### Comparison of learning schedule
Use ResNet101 as the backbone

share the common learning rate for the first and last epoch (lr_start = 1e-3, lr_end = 1e-6)

| Learning<br>Schedule           | Hand  | Target | Hand + <br>Side | Hand + <br>Contact State | Hand + <br>Target | Hand + <br>All |
|--------------------------------|-------|--------|-----------------|--------------------------|-------------------|----------------|
| decay_epoch = 1<br>epochs = 4  | 80.75 | 46.02  | 65.68           | 48.86                    | 19.29             | 16.08          |
| decay_epoch = 2<br>epochs = 7  | 81.13 | 51.49  | 68.20           | 54.46                    | 28.40             | 21.73          |
| decay_epoch = 3<br>epochs = 10 | 81.35 | 52.61  | 68.51           | 55.50                    | 30.85             | 28.18          |


### Comparison of network optimisation

parameter setting
  - ResNet101 as the backbone
  - lr_start = 1e-3, lr_end = 1e-5
  - lr_decay_epoch = 3, epochs = 10




## Citation

If this work is helpful in your research, please cite:
```
@INPROCEEDINGS{Shan20, 
    author = {Shan, Dandan and Geng, Jiaqi and Shu, Michelle  and Fouhey, David},
    title = {Understanding Human Hands in Contact at Internet Scale},
    booktitle = CVPR, 
    year = {2020} 
}
```
When you use the model trained on our ego data, make sure to also cite the original datasets ([Epic-Kitchens](https://epic-kitchens.github.io/2018), [EGTEA](http://cbs.ic.gatech.edu/fpv/) and [CharadesEgo](https://prior.allenai.org/projects/charades-ego)) that we collect from and agree to the original conditions for using that data.
