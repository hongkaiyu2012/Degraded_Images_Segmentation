############################################## 
Image Semantic Segmentation in PyTroch. 

Original Code Link: https://github.com/guo2004131/Degraded_Images_Segmentation 

Motified by hongkaiyu2012

05/12/2019
##############################################

1. Required packages: 

numpy, scipy, matplotlib, pytorch, PIL, python 2.7, torchfcn

2. Dataset format: see CamVid and Custom as an example, you need to follow the same data format. 

3. In this code, only FCN8S and SegNet is given. You can define more networks or your own networks in the folder 'models'. 

4. Training FCN8s: python train_fcn8s_atonce.py 

5. Testing FCN8s: python test_fcn8s_atonce.py -m PATH_TO_TRAINED_MODEL

Note: You need to define dataset and datasetroot in training and testing. Custom means your own dataset

example:

python test_fcn8s_atonce.py -m /home/hongkai/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_MAX_ITERATION-100000_LR-1e-10_INTERVAL_VALIDATE-4000_WEIGHT_DECAY-0.0005_MOMENTUM-0.99_VCS-4cc1bc1_TIME-20190426-144408/model_best.pth.tar

If you use PyCharm, you need to define -m PATH_TO_TRAINED_MODEL in Parameters of Edit Configurations of Run, then directly run test_fcn8s_atonce.py. 



