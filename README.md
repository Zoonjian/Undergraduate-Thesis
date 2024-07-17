## MSAM Deeplabv3+：A Multi-Scale Fusion Module And Coordinate Attention Mechanism Based Semantic Segmentation Algorithm
Design a multi-scale weighted fusion module and introduce a coordinate attention mechanism to improve the Deeplabv3+

### Train
1. put the label file in the SegmentClass folder under VOC2007 in the VOCdevkit folder
2. put the image files in the JPEGImages folder under the VOC2007 folder in the VOCdevkit folder
3. generate the corresponding txt file using the VOC_ annotation.py file before training
4. select the backbone model and downsampling factor you want to use in the train.py folder
5. modify the num_classes in train.py to the number of categories+1
6. run train.py

### Test  
1. modify model_math, num_classes, and backbone in the deeplab.py to correspond to the trained file Model Path corresponds to the weight files in the logs folder, num_classes represents the number of classes to be predicted+1, and backbone is the backbone feature extraction network used
2. run predict.py : python img_path

### Evluation
1. Set num_classes in get-miou.py to the number of predicted classes+1
2. Set the name_classes in get-miou.cpy to the categories that need to be distinguished
3. To obtain the size of miou, run get_miou.py
